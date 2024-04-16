import torch

class MetricComputer:
    def __init__(self, total_samples=None):
        self.total_samples = total_samples
        self.correct = 0
        self.loss = 0.0 

    def calculate_correct_preds(self, labels, logits):
        logits = torch.sigmoid(logits)
        label_indices = torch.argmax(labels, axis=-1)
        pred_indices = torch.argmax(logits, axis=-1)
        num_corrects = torch.sum(label_indices == pred_indices)
        self.correct+= num_corrects
        return num_corrects

    def compute_f1_score(corrects, totals):
        pass
