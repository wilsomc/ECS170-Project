'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score

from code.base_class.evaluate import evaluate
from sklearn.metrics import precision_score, f1_score, precision_score, recall_score

class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        acc = accuracy_score(true_y, pred_y)
        prec = precision_score(true_y, pred_y, average='macro', zero_division=0)
        rec = recall_score(true_y, pred_y, average='macro', zero_division=0)
        f1 = f1_score(true_y, pred_y, average='macro', zero_division=0)

        print(f'-- Accuracy: {acc}')
        print(f'-- Precision (macro): {prec}')
        print(f'-- Recall (macro): {rec}')
        print(f'-- F1 Score (macro): {f1}')

        return acc
