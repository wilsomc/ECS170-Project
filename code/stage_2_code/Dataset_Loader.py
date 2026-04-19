'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.dataset import dataset

import pandas as pd

class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_train_source_file_name = None
    dataset_test_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading stage_2 data...')
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        df_train = pd.read_csv(self.dataset_source_folder_path + self.dataset_train_source_file_name, header=None)
        df_test = pd.read_csv(self.dataset_source_folder_path + self.dataset_test_source_file_name, header=None)
        dnp_train = df_train.to_numpy()
        dnp_test = df_test.to_numpy()

        for i in range(len(dnp_train)):
            elements = dnp_train[i].tolist()
            X_train.append(elements[1:])  # Features
            y_train.append(elements[0])  # Labels

        for i in range(len(dnp_test)):
            elements = dnp_test[i].tolist()
            X_test.append(elements[1:])  # Features
            y_test.append(elements[0])  # Labels

        print(X_train)

        return {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}

if __name__ == '__main__':
    data_obj = Dataset_Loader('stage_two_train', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_train_source_file_name = 'train.csv'
    data_obj.dataset_test_source_file_name = 'test.csv'

    data_obj.load()


