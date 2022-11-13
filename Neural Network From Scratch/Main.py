from fullyconnected import FullLayer
from softmax import SoftMaxLayer
from crossEntropy import CrossEntropyLayer
from Sequential import Sequential
from relu import ReluLayer
from dataset import cifar100
import numpy as np
import matplotlib.pyplot as plt
import warnings
import csv

warnings.filterwarnings('ignore')
class runSequential():
    def __init__(self):
        self.layer1 = FullLayer(32 * 32 * 3, 500)
        self.relu = ReluLayer()
        self.layer2 = FullLayer(500, 5)
        self.softmax = SoftMaxLayer()
        self.loss = CrossEntropyLayer()
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None

        self.model = Sequential(
            (self.layer1,
             self.relu,
             self.layer2,
             self.softmax),
            self.loss
        )

    def initialize_data(self):
        (self.xtrain, self.ytrain), (self.xtest, self.ytest) = cifar100(201802130)

    def run(self):
        batch_size = 100
        epochs = np.arange(1, 21, step=1)
        lrs = np.array([0.01, 0.1, 0.3])
        losses = []
        data_l = []
        for i in lrs:
            data_temp = []
            model = runSequential()
            model.initialize_data()
            losses.append(self.model.fit(self.xtrain, self.ytrain, lr=i, epochs=epochs.size, batch_size=batch_size))
            y_predicted = self.model.predict(self.xtest, batch_size)
            data_temp.append(i)
            data_temp.append(np.mean(y_predicted == self.ytest))
            data_l.append(data_temp)
        for i in range(len(losses)):
            plt.plot(epochs, losses[i], label='Training Loss for Learning Rate = ' + str(lrs[i]))
        plt.legend()
        plt.show()
        with open('acc_results.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_l)


model_test = runSequential()
model_test.initialize_data()
model_test.run()
