'''
Concrete IO class for a specific dataset
'''

import pickle
import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from code.stage_3_code.CustomDataset import CustomDataset

class Dataset_Loader():
    dataset_name = None
    dataset_description = None
    setType = None

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None, dSetType=None):
        self.dataset_name = dName
        self.dataset_description = dDescription
        self.setType = dSetType

    def load(self):
        print("Loading stage_3 data...")
        f = open(f'{self.dataset_source_folder_path}/{self.dataset_source_file_name}', 'rb')
        data = pickle.load(f)
        f.close()

        transform = None
        if self.setType == "cifar":
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        elif self.setType == "mnist" or self.setType == "orl":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))])
        else:
            raise ValueError("Not a valid setType. Options are: cifar, orl, mnist")

        batch_size = 32

        trainset = CustomDataset(data['train'], transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = CustomDataset(data['test'], transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

        return trainloader, testloader


