import numpy
import mxnet


class MSE():
    def __init__(self):
        self.sum_metric = 0
        self.num_inst = 0

    # it is expected that the shape is num*c*h*w
    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            temp = mxnet.ndarray.sum(((label - pred) ** 2.0))
            self.sum_metric += temp.asnumpy()
            self.num_inst += label.shape[0]

    def get(self):

        return 'MSE', self.sum_metric/self.num_inst

    def reset(self):
        self.sum_metric = 0.0
        self.num_inst = 0
