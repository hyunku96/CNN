import pickle, gzip
import numpy as np
from tqdm import *

class mlp:
    def __init__(self, lr=0.01):
        #open dataset
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
        #2-layer fc weights
        self.w1 = np.random.rand(28 * 28, 28) - 0.5
        self.b1 = np.random.rand(28) - 0.5
        self.w2 = np.random.rand(28, 10) - 0.5
        self.b2 = np.random.rand(10) - 0.5
        #fc layer1 output
        self.o1 = None
        #fc layer1 activated ouput
        self.a1 = None
        #fc layer2 output
        self.o2 = None
        #fc layer2 activated output
        self.a2 = None
        #learning rate
        self.lr = lr

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, index):
        img = self.train_set[0][index]
        self.o1 = np.dot(img, self.w1) + self.b1
        self.a1 = self.sigmoid(self.o1)
        self.o2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.o2)

    def backward(self, index):
        # one-hot
        label = np.zeros(10)
        label[self.train_set[1][index]] = 1
        #back propagation
        dLda2 = self.a2 - label #10, 1
        da2do2 = self.sigmoid(self.o2) * (1 - self.sigmoid(self.o2)) #10, 1
        do2da1 = self.w2 #10 * 28
        da1do1 = self.sigmoid(self.o1) * (1 - self.sigmoid(self.o1)) #28, 1
        do1dw1 = self.train_set[0][index] #784 * 1
        #calculate gradients
        db2 = dLda2 * da2do2 #10, 1
        dw2 = np.dot(self.a1.reshape(-1, 1), db2.reshape(1, -1)) #28, 10
        db1 = np.dot(self.w2, db2) * da1do1 #28, 1
        dw1 = np.dot(self.train_set[0][index].reshape(-1, 1), db1.reshape(1, -1)) #784, 28
        self.w1 = self.w1 - self.lr * dw1
        self.b1 = self.b1 - self.lr * db1
        self.w2 = self.w2 - self.lr * dw2
        self.b2 = self.b2 - self.lr * db2
        #return loss value(Mean Squared Error
        return np.sum(dLda2**2) / len(dLda2) * 2

    def train(self, epoch=100):
        for i in range(epoch):
            loss = 0
            #rearrange indexes
            indexes = np.random.permutation(len(self.train_set[1]))
            for index in tqdm(indexes):
                self.forward(index)
                loss = loss + self.backward(index)
            print("epoch:{0}, loss:{1}, accuracy:{2}".format(i, loss / len(indexes), self.validation()))

    def validation(self):
        accuracy = 0
        for i in range(len(self.valid_set[1])):
            #forward calculation
            img = self.valid_set[0][i]
            self.o1 = np.dot(img, self.w1) + self.b1
            self.a1 = self.sigmoid(self.o1)
            self.o2 = np.dot(self.a1, self.w2) + self.b2
            self.a2 = self.sigmoid(self.o2)
            if self.a2.argmax() == self.valid_set[1][i]:
                accuracy = accuracy + 1
        #return validation accuracy
        return accuracy / len(self.valid_set[1])

    def test(self):
        accuracy = 0
        for i in range(len(self.test_set[1])):
            img = self.test_set[0][i]
            self.o1 = np.dot(img, self.w1) + self.b1
            self.a1 = self.sigmoid(self.o1)
            self.o2 = np.dot(self.a1, self.w2) + self.b2
            self.a2 = self.sigmoid(self.o2)
            if self.a2.argmax() == self.test_set[1][i]:
                accuracy = accuracy + 1
        print("accuracy:{0}".format(accuracy / len(self.test_set[1])))

m = mlp()
m.train()
m.test()