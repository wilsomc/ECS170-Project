'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import KFold
import numpy as np

class Setting_KFold_CV(setting):
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()

        score_list = []
        fold_count = 1
        print('************ Fold:', fold_count, '************')
        X_train, X_test = np.array(loaded_data['X_train']), np.array(loaded_data['X_test'])
        y_train, y_test = np.array(loaded_data['y_train']), np.array(loaded_data['y_test'])

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

        