import sys
import os
# Insert project root at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from code.stage_3_code.Method_CNN_CIFAR import Method_CNN_CIFAR
# from code.stage_3_code.Method_CNN import Method_CNN_ORL
# from code.stage_3_code.Method_CNN import Method_CNN_MNIST
from code.stage_3_code.Dataset_Loader import Dataset_Loader

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

#--- Convolution Neural Network Script ---
if __name__ == '__main__':
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    # ---- object initialization section ---------------
    # Define setType (the type of dataset) in Dataset_Loader for transformer
    data_obj = Dataset_Loader(dName='dataset for cifar-10', dDescription='train/test set', dSetType="cifar")
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'CIFAR'

    method_obj = Method_CNN_CIFAR(nName='convolution neural network', mDescription='using cifar-10 dataset', loaded_data=data_obj.load())

    # ---- running section ---------------------------------
    print('************ Start (CIFAR-10 CNN) ************')
    method_obj.run()
    print('************ Finish ************')
    # ------------------------------------------------------