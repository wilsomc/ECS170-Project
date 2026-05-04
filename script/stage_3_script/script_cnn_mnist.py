import sys
import os
# Insert project root at the front of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
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
    data_obj = Dataset_Loader(dName='dataset for orl', dDescription='train/test set')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'ORL'

    method_obj = Method_CNN_ORL(nName='convolution neural network', mDescription='using orl dataset', loaded_data=data_obj.load())

    # ---- running section ---------------------------------
    print('************ Start (Original) ************')
    method_obj.run()
    print('************ Finish ************')
    # ------------------------------------------------------