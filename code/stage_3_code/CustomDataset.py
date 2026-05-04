'''
Custom dataset for torch.vision iterator
'''

import torch
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Extract image and label from the dictionary
        instance = self.data_list[idx]
        image_matrix = instance['image']
        label = instance['label']

        # Ensure the image is a numpy array (often required before transforming)
        if not isinstance(image_matrix, np.ndarray):
            image_matrix = np.array(image_matrix)

        # Apply transformations (like converting to a PyTorch Tensor)
        if self.transform:
            image_tensor = self.transform(image_matrix)
        else:
            # Fallback if no transform is provided (less ideal)
            image_tensor = torch.tensor(image_matrix, dtype=torch.float32)

        # Convert label to a PyTorch tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor