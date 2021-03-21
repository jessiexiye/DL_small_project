# Matplotlib
import matplotlib.pyplot as plt
# Numpy
import numpy as np
# Pillow
from PIL import Image
# Torch
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms

class Infection_Dataset(Dataset):

    def __init__(self, dataset_group):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.

        - dataset_group should take values in 'train', 'test' or 'val'.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'infected'}
        
        # The dataset consists only of training images
        self.dataset_group = dataset_group
        
        # dataset_numbers: number of images in each part of the dataset
        # datset_paths: path to images for different parts of the dataset
        if dataset_group == 'train':
            self.dataset_numbers = {'train_normal': 1341,\
                                    'train_infected_noncovid': 2530,\
                                    'train_infected_covid': 1345}
            self.dataset_paths = {'train_normal': './dataset/train/normal/',\
                                  'train_infected_noncovid': './dataset/train/infected/non-covid/',\
                                  'train_infected_covid': './dataset/train/infected/covid/'}

        elif dataset_group == 'val':
            self.dataset_numbers = {'val_normal': 8,\
                                    'val_infected_noncovid': 8,\
                                    'val_infected_covid': 8}
            self.dataset_paths = {'val_normal': './dataset/val/normal/',\
                                  'val_infected_noncovid': './dataset/val/infected/non-covid/',\
                                  'val_infected_covid': './dataset/val/infected/covid/'}

        elif dataset_group == 'test':
            self.dataset_numbers = {'test_normal': 234,\
                                    'test_infected_noncovid': 242,\
                                    'test_infected_covid': 138}
            self.dataset_paths = {'test_normal': './dataset/test/normal/',\
                                  'test_infected_noncovid': './dataset/test/infected/non-covid/',\
                                  'test_infected_covid': './dataset/test/infected/covid/'}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the {} dataset of the Lung Dataset".format(self.dataset_group)
        msg += " used for the Small Project in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected_covid' or 'infected_noncovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        group_val = self.dataset_group

        # Asserts checking for consistency in passed parameters
        
        max_val = sum(self.dataset_numbers.values())
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)

        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im

    def show_img(self, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - class_val variable should be set to 'normal', 'infected_covid' or 'infected_noncovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        print(self.dataset_group)
        im = self.open_img(self.dataset_group, class_val, index_val)
        
        # Display
        plt.imshow(im)
        return plt
    
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1])
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0])
        elif index - first_val < second_val:
            index = index - first_val
            class_val = 'infected_noncovid'
            label = torch.Tensor([0, 1])
        else:
            class_val = 'infected_covid'
            index = index - first_val - second_val
            label = torch.Tensor([0, 1])
        im = self.open_img(self.dataset_group, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label


class Covid_Dataset(Dataset):

    def __init__(self, dataset_group):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.

        - dataset_group should take values in 'train', 'test' or 'val'.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (covid and non-covid)
        self.classes = {0: 'covid', 1: 'non-covid'}
        
        # The dataset consists only of training images
        self.dataset_group = dataset_group
        
        # dataset_numbers: number of images in each part of the dataset
        # datset_paths: path to images for different parts of the dataset
        if dataset_group == 'train':
            self.dataset_numbers = {'train_infected_noncovid': 2530,\
                                    'train_infected_covid': 1345}
            self.dataset_paths = {'train_infected_noncovid': './dataset/train/infected/non-covid/',\
                                  'train_infected_covid': './dataset/train/infected/covid/'}

        elif dataset_group == 'val':
            self.dataset_numbers = {'val_infected_noncovid': 8,\
                                    'val_infected_covid': 8}
            self.dataset_paths = {'val_infected_noncovid': './dataset/val/infected/non-covid/',\
                                  'val_infected_covid': './dataset/val/infected/covid/'}

        elif dataset_group == 'test':
            self.dataset_numbers = {'test_infected_noncovid': 242,\
                                    'test_infected_covid': 138}
            self.dataset_paths = {'test_infected_noncovid': './dataset/test/infected/non-covid/',\
                                  'test_infected_covid': './dataset/test/infected/covid/'}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the {} dataset of the Lung Dataset".format(self.dataset_group)
        msg += " used for the Small Project in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'infected_covid' or 'infected_noncovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        group_val = self.dataset_group

        # Asserts checking for consistency in passed parameters
        
        max_val = sum(self.dataset_numbers.values())
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)

        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im

    def show_img(self, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - class_val variable should be set to 'infected_covid' or 'infected_noncovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        print(self.dataset_group)
        im = self.open_img(self.dataset_group, class_val, index_val)
        
        # Display
        plt.imshow(im)
        return plt
    
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        if index < first_val:
            class_val = 'infected_noncovid'
            label = torch.Tensor([1, 0])
        else:
            class_val = 'infected_covid'
            index = index - first_val
            label = torch.Tensor([0, 1])
        im = self.open_img(self.dataset_group, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label


class Multiclass_Dataset(Dataset):

    def __init__(self, dataset_group):
        """
        Constructor for generic Dataset class - simply assembles
        the important parameters in attributes.

        - dataset_group should take values in 'train', 'test' or 'val'.
        """
        
        # All images are of size 150 x 150
        self.img_size = (150, 150)
        
        # Only two classes will be considered here (normal and infected)
        self.classes = {0: 'normal', 1: 'covid', 2: 'non-covid'}
        
        # The dataset consists either training, test or val set
        self.dataset_group = dataset_group
        
        # dataset_numbers: number of images in each part of the dataset
        # datset_paths: path to images for different parts of the dataset
        if dataset_group == 'train':
            self.dataset_numbers = {'train_normal': 1341,\
                                    'train_infected_noncovid': 2530,\
                                    'train_infected_covid': 1345}
            self.dataset_paths = {'train_normal': './dataset/train/normal/',\
                                  'train_infected_noncovid': './dataset/train/infected/non-covid/',\
                                  'train_infected_covid': './dataset/train/infected/covid/'}

        elif dataset_group == 'val':
            self.dataset_numbers = {'val_normal': 8,\
                                    'val_infected_noncovid': 8,\
                                    'val_infected_covid': 8}
            self.dataset_paths = {'val_normal': './dataset/val/normal/',\
                                  'val_infected_noncovid': './dataset/val/infected/non-covid/',\
                                  'val_infected_covid': './dataset/val/infected/covid/'}

        elif dataset_group == 'test':
            self.dataset_numbers = {'test_normal': 234,\
                                    'test_infected_noncovid': 242,\
                                    'test_infected_covid': 138}
            self.dataset_paths = {'test_normal': './dataset/test/normal/',\
                                  'test_infected_noncovid': './dataset/test/infected/non-covid/',\
                                  'test_infected_covid': './dataset/test/infected/covid/'}

    def describe(self):
        """
        Descriptor function.
        Will print details about the dataset when called.
        """
        
        # Generate description
        msg = "This is the {} dataset of the Lung Dataset".format(self.dataset_group)
        msg += " used for the Small Project in the 50.039 Deep Learning class"
        msg += " in Feb-March 2021. \n"
        msg += "It contains a total of {} images, ".format(sum(self.dataset_numbers.values()))
        msg += "of size {} by {}.\n".format(self.img_size[0], self.img_size[1])
        msg += "The images are stored in the following locations "
        msg += "and each one contains the following number of images:\n"
        for key, val in self.dataset_paths.items():
            msg += " - {}, in folder {}: {} images.\n".format(key, val, self.dataset_numbers[key])
        print(msg)

    def open_img(self, group_val, class_val, index_val):
        """
        Opens image with specified parameters.
        
        Parameters:
        - group_val should take values in 'train', 'test' or 'val'.
        - class_val variable should be set to 'normal', 'infected_covid' or 'infected_noncovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        
        Returns loaded image as a normalized Numpy array.
        """
        
        group_val = self.dataset_group

        # Asserts checking for consistency in passed parameters
        
        max_val = sum(self.dataset_numbers.values())
        err_msg = "Error - index_val variable should be an integer between 0 and the maximal number of images."
        err_msg += "\n(In {}/{}, you have {} images.)".format(group_val, class_val, max_val)
        assert isinstance(index_val, int), err_msg
        assert index_val >= 0 and index_val <= max_val, err_msg
        
        # Open file as before
        path_to_file = '{}{}.jpg'.format(self.dataset_paths['{}_{}'.format(group_val, class_val)], index_val)

        with open(path_to_file, 'rb') as f:
            im = np.asarray(Image.open(f))/255
        f.close()
        return im

    def show_img(self, class_val, index_val):
        """
        Opens, then displays image with specified parameters.
        
        Parameters:
        - class_val variable should be set to 'normal', 'infected_covid' or 'infected_noncovid'.
        - index_val should be an integer with values between 0 and the maximal number of images in dataset.
        """
        
        # Open image
        print(self.dataset_group)
        im = self.open_img(self.dataset_group, class_val, index_val)
        
        # Display
        plt.imshow(im)
        return plt
    
    def __len__(self):
        """
        Length special method, returns the number of images in dataset.
        """
        
        # Length function
        return sum(self.dataset_numbers.values())

    def __getitem__(self, index):
        """
        Getitem special method.
        
        Expects an integer value index, between 0 and len(self) - 1.
        
        Returns the image and its label as a one hot vector, both
        in torch tensor format in dataset.
        """
        
        # Get item special method
        first_val = int(list(self.dataset_numbers.values())[0])
        second_val = int(list(self.dataset_numbers.values())[1])
        if index < first_val:
            class_val = 'normal'
            label = torch.Tensor([1, 0, 0])
        elif index - first_val < second_val:
            index = index - first_val
            class_val = 'infected_noncovid'
            label = torch.Tensor([0, 1, 0])
        else:
            class_val = 'infected_covid'
            index = index - first_val - second_val
            label = torch.Tensor([0, 0, 1])
        im = self.open_img(self.dataset_group, class_val, index)
        im = transforms.functional.to_tensor(np.array(im)).float()
        return im, label