'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Evaluate_Accuracy():
    all_predicted = []
    all_labels = []

    def __init__(self, dPredicted, dLabels):
        self.all_predicted = dPredicted
        self.all_labels = dLabels
    
    def evaluate(self):
        print('evaluating performance...')

        acc = accuracy_score(self.all_labels, self.all_predicted)
        prec = precision_score(self.all_labels, self.all_predicted, average='macro', zero_division=0)
        rec = recall_score(self.all_labels, self.all_predicted, average='macro', zero_division=0)
        f1 = f1_score(self.all_labels, self.all_predicted, average='macro', zero_division=0)

        print(f'-- Accuracy: {acc * 100:.2f}')
        print(f'-- Precision (macro): {prec * 100:2f}')
        print(f'-- Recall (macro): {rec * 100:2f}')
        print(f'-- F1 Score (macro): {f1 * 100:2f}')
