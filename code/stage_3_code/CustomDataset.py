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

        # Detect if labels are 1-indexed (e.g., ORL dataset labels 1-40)
        # CrossEntropyLoss expects 0-indexed labels
        labels = [instance['label'] for instance in data_list]
        self.min_label = min(labels)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Extract image and label from the dictionary
        instance = self.data_list[idx]
        image_matrix = instance['image']
        label = instance['label']

        # Shift labels to 0-indexed if needed (e.g., ORL: 1-40 -> 0-39)
        label = label - self.min_label

        # Ensure the image is a numpy array (often required before transforming)
        if not isinstance(image_matrix, np.ndarray):
            image_matrix = np.array(image_matrix)

        # Ensure correct dtype for transforms.ToTensor()
        # ToTensor expects uint8 for HxW or HxWxC images
        if image_matrix.dtype == np.float64:
            image_matrix = image_matrix.astype(np.float32)

        # Apply transformations (like converting to a PyTorch Tensor)
        if self.transform:
            image_tensor = self.transform(image_matrix)
        else:
            # Fallback if no transform is provided (less ideal)
            image_tensor = torch.tensor(image_matrix, dtype=torch.float32)

        # Convert label to a PyTorch tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor