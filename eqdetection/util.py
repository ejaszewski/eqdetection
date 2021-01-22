import torch

class Statistics():
    def __init__(self, device=None):
        self.true_pos = torch.tensor(0.0, device=device)
        self.false_pos = torch.tensor(0.0, device=device)
        self.false_neg = torch.tensor(0.0, device=device)

    def add_pred_actual(self, pred, actual):
        true_pos = pred * actual
        false_pos = pred * ~actual
        false_neg = ~pred * actual

        self.true_pos += true_pos.sum()
        self.false_pos += false_pos.sum()
        self.false_neg += false_neg.sum()

    def add_corr_actual(self, corr, actual):
        true_pos = corr * actual
        false_pos = ~corr * ~actual
        false_neg = ~corr * actual

        self.true_pos += true_pos.sum()
        self.false_pos += false_pos.sum()
        self.false_neg += false_neg.sum()

    def log(self, writer, prefix, step):
        writer.add_scalar(f'{prefix}/Precision', self.precision, step)
        writer.add_scalar(f'{prefix}/Recall', self.recall, step)
        writer.add_scalar(f'{prefix}/F-1', self.f1, step)

    @property
    def precision(self):
        denom = self.true_pos + self.false_pos
        if denom.item() == 0.0:
            return denom
        return self.true_pos / denom
    
    @property
    def recall(self):
        denom = self.true_pos + self.false_neg
        if denom.item() == 0.0:
            return denom
        return self.true_pos / denom
    
    @property
    def f1(self):
        precision = self.precision
        recall = self.recall
        denom = precision + recall
        if denom.item() == 0.0:
            return denom
        return 2 * (precision * recall) / denom