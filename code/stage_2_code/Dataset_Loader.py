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
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        df = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name, header=None)
        dnp = df.to_numpy()

        for i in range(len(dnp)):
            elements = dnp[i].tolist()
            X.append(elements[1:])  # Features
            y.append(elements[0])  # Labels

        return {'X': X, 'y': y}

