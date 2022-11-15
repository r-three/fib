import numpy as np

class Scorer(object):

    def __init__(self, metric):
        self.metric = metric

        if metric == "multiple_choice":
            self.total_numCorrectDatapoints = 0
            self.total_numDatapoints = 0
        else:
            raise ValueError(f"Invalid metric {metric}")

    def add_batch(self, batchOf_evalInfo):
        pred_choice = np.asarray(batchOf_evalInfo["pred_choice"])
        lbl = np.asarray(batchOf_evalInfo["lbl"])

        which_datapointsCorrect = pred_choice == lbl
        self.total_numCorrectDatapoints += np.sum(which_datapointsCorrect)
        self.total_numDatapoints += which_datapointsCorrect.shape[0]

        return which_datapointsCorrect.tolist()

    def get_score(self):
        mulChoice_acc = float(round(self.total_numCorrectDatapoints / self.total_numDatapoints,3))
        return {"multiple-choice-accuracy": mulChoice_acc}


