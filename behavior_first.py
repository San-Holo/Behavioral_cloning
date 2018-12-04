import numpy as np
import pandas as pd
from random import shuffle
from math import floor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from skimage import io
from glob import iglob

from PIL import Image

from functools import partial

import itertools as it

from tqdm import tqdm

import imgaug
from imgaug import augmenters as aug

####Utils####

np.random.seed(50)
torch.manual_seed(50)

def load_images(dirpath):
    """Load images from dirpath.

    Returns
    --------
        Tensor with all images, size : (nb_images, height, width, channels)
    """
    preprocess = transforms.Compose([transforms.ToTensor()])
    images = np.array(list(map(io.imread, iglob(dirpath + "*png")))) / 255
    images = images[..., :3]
    images = np.swapaxes(images, 1, 3)
    images = np.swapaxes(images, 2, 3) #Good order to channel - height - width
    images = torch.tensor(images)
    images = (images - images.mean()) / images.std()
    return images, images.size()[1:]


def load_labels(filepath):
    """Load labels from filepath (csv file)

    Returns
    --------
        Tensor of shape (nb_example, 2). On second dim, first is throttling factor (< 0 for braking), second is steering's
    """
    df = pd.read_csv(filepath)
    throttling = torch.tensor(df["throttle"].values - df["brake"].values).unsqueeze(1)
    steering = torch.tensor(df["steering"].values).unsqueeze(1)
    labels = torch.cat([throttling, steering], 1)
    return labels


def one_hot_bin_encoder(speeds, bins):
    """One hot encoding for speed values.

    Parameters
    -----------
    speeds : numpy array or dataframe
        Structure containing speed values

    bins : int
        Number of bins used to discretize speed.

    Returns
    --------
        Encoded values of speed
    """
    speeds_ = np.repeat(speeds[..., np.newaxis], len(bins), 0)
    ids = (speeds_ < bins).argmax(1)
    encoded = np.eye(len(bins))[ids]
    return encoded


def load_speed(filepath, encoder=(lambda x: x)):
    """Load speed from filepath and applying encoder.

    Parameters
    -----------
        filepath : str
            The filepath

        encoder : function
            Function to call so speed is encoded as wished (one-hot for example)

    Returns
    --------
        Tensor of size (nb_examples, n), where n is the dimension given by the encoder
    """
    df = pd.read_csv(filepath)
    speed = df["speed"].values
    speed = encoder(speed)
    speed = torch.tensor(speed)
    return speed


class ADLDataset(Dataset):
    """A class to create our own dataset which can be used with DataLoader from torch"""

    def __init__(self, dir_path, csv_path, bins):
        """Loading and assigning data in order to create dataset

        Parameters
        ----------
        dir_path :  String
            A path to the directory in which pictures are stored

        csv_path :  String
            A path to the file in which data for one user are stored

        bins : int
            Number of bins used to discretize speed.
        """
        encoder = partial(one_hot_bin_encoder, bins = bins)

        self.images, self.img_size = load_images(dir_path)
        self.speed = load_speed(csv_path, encoder)
        self.labels = load_labels(csv_path)
        self.speed_size = len(bins)

    def __getitem__(self, index):
        """Loading and assigning data in order to create dataset

        Parameters
        ----------
        index :  int
            Nth data in the dataset

        Returns
        ---------
        x : (tensor, int) tuple
            Entries for learning -> (img, speed)

        y : (float, float) tuple
            Labels -> (throttle, steering). Note : throttle is calculating using brake in the original csv file.
        """

        image_tensor = self.images[index]
        speed = self.speed[index]
        labels = self.labels[index]

        return (image_tensor, speed), labels

    def __len__(self):
        """Returning number of datas"""
        return len(self.images)



####Model####

def conv_block(in_filter, output_filter, nb_conv, kernel_size=3, activation_function=nn.ReLU()):
    """To simplify the creation of convolutional sequences

    Parameters
    ----------
    in_filter :  int
        Number of filters that we want in entry

    output_filter :  int
        Number of filters that we want in output

    nb_conv : int
        Number of convolution layers

    activation_function : nn Function
        Activation function after each convolution

    Returns
    ---------
    sequential : Sequential torch Object
        The convolutional sequence that we were seeking
    """
    nbchannel = in_filter
    nbfilter = output_filter
    sequential = []

    for i in range(nb_conv):
            sequential.append(nn.Conv2d(nbchannel, nbfilter, kernel_size, padding=1))
            sequential.append(activation_function)
            nbchannel = nbfilter
    sequential.append(nn.MaxPool2d(2))
    return sequential

def network_from_shape(net_structure, activation=nn.ReLU()):
    """To simplify the creation of fully connected layers sequences

    Parameters
    ----------
    net structure: int list
        Describe each layer size -> one entry of the list is a layer conv_size

    activation_function : nn Function
        Activation function after each layer of the net

    Returns
    ---------
    temp :  Torch object list
        The fully connected sequence with the last activation function "tanh"
    """
    temp = []
    for prev, next in zip(net_structure[:-1], net_structure[1:]):
         temp.append(nn.Linear(prev, next))
         temp.append(activation)
    temp = temp[:-1] # Remove last activation return temp
    return temp

class Behavior_clone(nn.Module):
    """A class for behavior cloning, with both convolutional layers and fully connected ones"""

    def __init__(self, conv_sizes, img_shape, network_lin_shape, speed_size):
        """Creating network following the one presented NVIDIA paper

        Parameters
        ----------
        conv_sizes :  int tuple list
            A list containing a tuple for each convolutionnal layer -> (number of filters, number of convolutions layers, kernel size)

        img_shape : int couple
            A tuple describing image's height and with

        network_lin_shape : int tuple
            A tuple describing the size of each fully connected layer

        speed_size : int
            Size of speed one-hot encoding
        """
        super(Behavior_clone, self).__init__()

        conv_sizes = [(img_shape[0],)] + conv_sizes
        layers = it.chain(*[conv_block(prev[0], curr[0], curr[1], curr[2]) for prev, curr in zip(conv_sizes, conv_sizes[1:])])
        self.conv = nn.Sequential(*layers)
        
        img_flatten_size = img_shape[1:]
        
        #Use of formula in : https://pytorch.org/docs/stable/nn.html#torch.nn.MaxPool2d to calculate input size for first layer of the fully connected block
        for i in range(len(conv_sizes)-1):
            img_flatten_size = floor((img_flatten_size[0] + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1), floor((img_flatten_size[1] + 2 * 0 - 1 * (2 - 1) - 1) / 2 + 1)
        
        img_flatten_size = img_flatten_size[0] * img_flatten_size[1] * conv_sizes[-1][0]
        self.fc = network_from_shape((img_flatten_size + speed_size,) + network_lin_shape + (2,), activation=nn.ReLU())
        self.fc = nn.Sequential(*self.fc)
        

    def forward(self, img, speed):
        """Passing img and speed input as describe in NVIDIA architecture

        Parameters
        ----------
        img :  int tuple
            (Height, width, channels)

        speed : float Tensor
            One hot vector -> One hot because if we concatenate just one value to the flattened output ouf convolutional sequence, speed will be not really taken in account for training
        """
        output = self.conv(img)
        output = torch.cat((output.reshape(output.size(0),-1), speed), dim=1)
        output = self.fc(output)
        return output

####Train utils####


def train(net, nb_epoch, train_loader, test_loader, val_loader, optimizer, loss_function, device):

    print("====== HYPERPARAMETERS =======")
    print("epochs=", nb_epoch)
    print("learning_rate=", 0.001)
    print("=" * 30)

    final_train_loss = []
    final_test_loss = []

    for epoch in range(nb_epoch):

        ####Train####
        with tqdm(train_loader, bar_format="{l_bar}{bar}{n_fmt}/{total_fmt}, ETA:{remaining}{postfix}", ncols=80, desc="Epoch " + str(epoch)) as t:
            mean_loss, n = 0, 0
            for (img,speed), y in t:
                img = img.to(device)
                speed = speed.to(device)
                y = y.to(device)
                pred = net(img, speed)
                loss = loss_function(pred, y)

                n += 1
                mean_loss = ((n-1) * mean_loss + loss.tolist()) / n
                t.set_postfix({"train_loss": "{0:.3f}".format(mean_loss)})

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            final_train_loss.append(mean_loss)

        ####Test####
        total_test_loss = 0
        for (img, speed), y in test_loader:

            img = img.to(device)
            speed = speed.to(device)
            y = y.to(device)

            #Forward pass
            tmp_test = net(img, speed)
            test_loss = loss_function(tmp_test, y)

            total_test_loss += test_loss.item()

        mean_test_loss = total_test_loss / len(test_loader)
        final_test_loss.append(mean_test_loss)
        print("Test loss = {:.2f}".format(mean_test_loss))

    ####Validation####
    total_val_loss = 0
    for (img, speed), y in val_loader:

        img = img.to(device)
        speed = speed.to(device)
        y = y.to(device)

        tmp_val = net(img, speed)
        val_loss = loss_function(tmp_val, y)

        total_val_loss += val_loss.item()

    mean_val_loss = total_val_loss / len(val_loader)
    print("Validation_loss = {:.2f}".format(mean_val_loss))

    return final_train_loss, final_test_loss

####Core####

if __name__ == '__main__':

    print("Loading data...")

"""Data : scene_fpv -> contains all images, and circuit_cw_user. csv -> all corresponding values

    EXAMPLE
    #######
    
    tmp = pd.read_csv('circuit_cw_user.csv')['speed']
    bins = np.arange(tmp.min(), tmp.max())
    dataset = ADLDataset('scene_fpv/', 'circuit_cw_user.csv', bins)
    batch_size = 32
    split_train_rest = .8 # 0.8 to train and 0.2 to rest


    #create indexes for Dataloaders
    indexes = list(range(len(dataset)))
    shuffle(indexes)
    split_index_train_rest = int(split_train_rest * len(dataset))
    split_index_val_test = split_index_train_rest + floor((len(dataset) - split_index_train_rest) / 2)

    train_indexes = indexes[:split_index_train_rest]
    val_indexes = indexes[split_index_train_rest:split_index_val_test+1]
    test_indexes = indexes[split_index_val_test:]

    train_sampler = SubsetRandomSampler(train_indexes)
    val_sampler = SubsetRandomSampler(val_indexes)
    test_sampler = SubsetRandomSampler(test_indexes)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,
                batch_size=batch_size, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(dataset=dataset,
                batch_size=batch_size, sampler=val_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                batch_size=batch_size, sampler=test_sampler)

    print("Building net...")

    device = torch.device("cpu") #pass to "cpu" if you want to

    #Add as many convolutional blocks you want is the first argument list, same for the third one concerning fully connected ones
    BC = Behavior_clone([(8, 2, 3), (16, 2, 3), (32, 2, 3)], dataset.img_size, (512,256,64,32), dataset.speed_size).double().to(device)
    optimizer = optim.Adadelta(BC.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    print("Training, Testing and Validating...")

    a, b = train(BC, 300, train_loader, test_loader, val_loader, optimizer, loss_function, device)

    print("Saving model and train/test data...")

    df_save = pd.DataFrame(np.column_stack([a, b]), columns=['Train', 'Test'])
    df_save.to_csv('train_test_values_dataBC_Linear.csv', index=False)

    torch.save(BC.state_dict(), "Behavior_cloning_weights_Linear")

"""
