'''
Concrete IO class for a specific dataset
'''

import pickle
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from code.stage_3_code.CustomDataset import CustomDataset

class Dataset_Loader():
    dataset_name = None
    dataset_description = None

    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_description = dDescription

    def load(self):
        print("Loading stage_3 data...")
        f = open(f'{self.dataset_source_folder_path}/{self.dataset_source_file_name}', 'rb')
        data = pickle.load(f)
        f.close()

        # Determine number of channels from the first training image
        sample_image = data['train'][0]['image']
        if len(sample_image.shape) == 2:
            # Grayscale (e.g., MNIST): 1 channel
            num_channels = 1
        else:
            num_channels = sample_image.shape[2]  # e.g., 3 for RGB

        # Build normalization based on channel count
        if num_channels == 1:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        batch_size = 32

        trainset = CustomDataset(data['train'], transform=transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

        testset = CustomDataset(data['test'], transform=transform)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

        return trainloader, testloader
