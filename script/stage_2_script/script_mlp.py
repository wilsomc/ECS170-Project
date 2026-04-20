from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Method_MLP_1 import Method_MLP_1
from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Setting_KFold_CV import Setting_KFold_CV
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
if 1:
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)

    # Enter which method to run
    # "original" - 2 layer model
    # "improved" - 4 layer model
    METHOD_TO_RUN = "improved"
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    data_obj = Dataset_Loader('stage_two_set', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_train_source_file_name = 'train.csv'
    data_obj.dataset_test_source_file_name = 'test.csv'

    method_obj = Method_MLP('multi-layer perceptron', 'Original (2 layers)')
    method_obj_1 = Method_MLP_1('multi-layer perceptron', 'Improved (4 layers)')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    #setting_obj = Setting_KFold_CV('k fold cross validation', '')
    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    if (METHOD_TO_RUN == 'original'):
        print('************ Start (Original) ************')
        result_obj.fold_count = None
        setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        mean_score, std_score = setting_obj.load_run_save_evaluate()
        print('************ Overall Performance ************')
        print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
        print('************ Finish ************')

    elif (METHOD_TO_RUN == 'improved'):
        print('************ Start (Improved) ************')
        result_obj.fold_count = 1
        setting_obj.prepare(data_obj, method_obj_1, result_obj, evaluate_obj)
        setting_obj.print_setup_summary()
        mean_score, std_score = setting_obj.load_run_save_evaluate()
        print('************ Overall Performance ************')
        print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
        print('************ Finish ************')
    # ------------------------------------------------------
    

    