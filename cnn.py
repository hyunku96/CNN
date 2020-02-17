import pickle, gzip
import numpy as np
from tqdm import *

class cnn:
    '''
    adaptive stride
    im2col convolution <- use GPU!
    store weights
    batch mode
    batch normalize
    Adam optimizer
    split backpropagation method
    stack forward computation and backtracking? <- modulize(super class)
    dynamic variable <- modulize(class inherit)
    '''
    def __init__(self, kernel_size=3, kernel1num=5, kernel2num=10, pool_size=2, lr=0.01, img_size=28):
        # Load the dataset -> tuple, [0][index]:data, [1][index]:label
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.kernel1num = kernel1num
        self.kernel2num = kernel2num
        self.pool_size = pool_size
        # cnn weights
        self.k1 = np.random.rand(kernel1num, 1, kernel_size, kernel_size) - 0.5 #kernel num, img channel, kernel size**2
        self.kb1 = np.random.rand(kernel1num, img_size**2) - 0.5 #kernel num, output img size
        self.k2 = np.random.rand(kernel2num, kernel1num, kernel_size, kernel_size) - 0.5
        self.kb2 = np.random.rand(kernel2num, int(img_size / pool_size)**2) - 0.5
        # fc weights
        self.w1 = np.random.rand(int(img_size/pool_size/pool_size)**2 * kernel2num, int(img_size/pool_size/pool_size)**2) - 0.5  # input * hidden
        self.b1 = np.random.rand(int(img_size/pool_size/pool_size)**2) - 0.5
        self.w2 = np.random.rand(int(img_size/pool_size/pool_size)**2, 10) - 0.5  # hidden * output
        self.b2 = np.random.rand(10) - 0.5
        # fc outputs
        self.o1 = None
        self.a1 = None
        self.o2 = None
        self.a2 = None
        # max pooling output
        self.p1 = []
        self.p2 = []
        # conv ouput
        self.c1 = []  # store for backprop
        self.c2 = []
        # learning rate
        self.lr = lr

    def convolution(self, img, weight, bias, kernel_num):
        result = []
        padding = int(self.kernel_size / 2)
        img = np.reshape(img, (-1, self.img_size, self.img_size))  # depth, height, width
        img = np.pad(img, pad_width=padding, constant_values=(0))[padding:-1]
        k = weight
        kb = bias

        for n in range(kernel_num):
            out = []
            for height in range(padding, img.shape[1] - padding):
                for width in range(padding, img.shape[2] - padding):
                    tmp = 0
                    for depth in range(img.shape[0]):
                        area = img[depth, height - padding:height + padding + 1, width - padding:width + padding + 1]
                        tmp += np.sum(np.ravel(area, order='C') * np.ravel(k[n, depth], order='C'))
                    out.append(tmp)
            out = out + kb[n]
            result.append(np.reshape(out, (self.img_size, self.img_size)))

        return result
        # result's shape c1 = 5, 28, 28 / c2 = 10, 14, 14

    def ReLU(self, x):
        return np.maximum(0, x)

    def max_pooling(self, img):
        result = []
        self.img_size = int(self.img_size / 2)
        for depth in range(img.shape[0]):
            out = []
            for height in range(0, img.shape[1], self.pool_size):
                for width in range(0, img.shape[2], self.pool_size):
                    out.append(np.max(img[depth, height:height + self.pool_size, width:width + self.pool_size]))
            result.append(np.reshape(out, (self.img_size, self.img_size)))

        return result
        # p's shape p1 = 5, 14, 14 / p2 = 10, 7, 7

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, img):
        img = np.ravel(img, order='C')
        self.o1 = np.dot(img, self.w1) + self.b1
        self.a1 = self.sigmoid(self.o1)
        self.o2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.o2)

    def backward(self, index):
        # one-hot vector
        label = np.zeros(10)
        label[self.train_set[1][index]] = 1
        # back propagation with partial differentiation
        dLda2 = self.a2 - label  # 10, 1
        da2do2 = self.sigmoid(self.o2) * (1 - self.sigmoid(self.o2))  # 10, 1
        da1do1 = self.sigmoid(self.o1) * (1 - self.sigmoid(self.o1))  # 49, 1
        # calculate gradients
        db2 = dLda2 * da2do2  # 10, 1
        dw2 = np.dot(self.a1.reshape(-1, 1), db2.reshape(1, -1))  # 49, 10
        db1 = np.dot(self.w2, db2) * da1do1  # 49, 1
        dw1 = np.dot(np.reshape(self.p2, (-1, 1)), db1.reshape(1, -1))  # 490, 49
        # 2nd ReLU & maxpooling layer's gradient
        grad = np.dot(self.w1, db1) # 490, 1
        grad = grad.reshape(self.kernel2num, self.img_size, self.img_size)  # 10, 7, 7
        self.img_size = self.img_size * 2  # img_size = 14
        dp2dc2 = np.zeros((self.kernel2num, self.img_size, self.img_size))  # 10, 14, 14
        # make gradient map
        for depth in range(self.kernel2num):
            for height in range(0, self.img_size, self.pool_size):
                for width in range(0, self.img_size, self.pool_size):
                    area = self.c2[depth][height:height + self.pool_size, width:width + self.pool_size]
                    if area.any() > 0:  # if their exists positive values
                        maxloc = area.argmax()
                        # assign rear layer's gradient to maxpooling & ReLU layer's gradient
                        dp2dc2[depth, height + int(maxloc / 2), width + maxloc % 2] = grad[depth, int(height / 2), int(width / 2)]
        # 2nd convolution layer's gradient
        padding = int(self.kernel_size / 2)
        # calculate kernel's gradient
        img = np.pad(self.p1, pad_width=padding, constant_values=(0))[1:-1]
        dc2dk2 = np.array([])
        # convolution input & gradient to compute kernel's gradient
        for n in range(self.kernel2num):  # kernel num
            for depth in range(self.kernel1num):  # gradient's channel
                for height in range(self.kernel_size):  # height interval
                    for width in range(self.kernel_size):  # width interval
                        '''
                        tmp = 0
                        for kernel_height in range(dp2dc2.shape[1]):
                            for kernel_width in range(dp2dc2.shape[2]):
                                tmp += img[depth][height + kernel_height, width + kernel_width] * dp2dc2[n, kernel_height, kernel_width]
                        '''
                        area = img[depth][height:height+dp2dc2.shape[1], width:width+dp2dc2.shape[2]]
                        tmp = np.sum(np.ravel(area, order='C') * np.ravel(dp2dc2[n], order='C'))
                        dc2dk2 = np.append(dc2dk2, tmp)  # 10 * 5 * 3 * 3
        dc2dk2 = np.reshape(dc2dk2, (self.kernel2num, self.kernel1num, self.kernel_size, self.kernel_size)) # num, depth, height, width
        # convolution kernel(rotation 180') & gradient to compute input's gradient
        dc2dp1 = np.array([])
        for n in range(self.kernel2num):  # kernel num
            gradient = np.pad(dp2dc2[n], pad_width=padding, constant_values=(0))
            _k2 = np.rot90(self.k2[n], 2)

            tmp = []
            for depth in range(self.kernel1num):  # p1's depth
                for height in range(1, gradient.shape[0] - 1):
                    for width in range(1, gradient.shape[1] - 1):
                        area = gradient[height-1:height+2, width-1:width+2]
                        tmp.append(np.sum(np.ravel(area, order='C') * np.ravel(_k2[depth], order='C')))
            dc2dp1 = np.append(dc2dp1, tmp)
        dc2dp1 = np.reshape(dc2dp1, (self.kernel2num, self.kernel1num, self.img_size, self.img_size))
        dc2dp1 = dc2dp1.sum(axis=0) / self.kernel2num  # mean of each kernel's gradient
        # 1st ReLU & maxpooling layer's gradient
        self.img_size = self.img_size * 2  # img_size = 28
        dp1dc1 = np.zeros((self.kernel1num, self.img_size, self.img_size))  # 5, 28, 28
        # make gradient map
        for depth in range(self.kernel1num):
            for height in range(0, self.img_size, self.pool_size):
                for width in range(0, self.img_size, self.pool_size):
                    area = self.c1[depth][height:height + self.pool_size, width:width + self.pool_size]
                    if area.any() > 0:  # if their exists positive values
                        maxloc = area.argmax()
                        # assign rear layer's gradient to maxpooling & ReLU layer's gradient
                        dp1dc1[depth, height + int(maxloc / 2), width + maxloc % 2] = dc2dp1[depth, int(height / 2), int(width / 2)]
        # 1st convolution layer's gradient
        # calculate kernel's gradient
        img = np.reshape(self.train_set[0][index], (self.img_size, self.img_size))
        img = np.pad(img, pad_width=padding, constant_values=(0))
        dc1dk1 = np.array([])
        for n in range(self.kernel1num):  # kernel num
            for height in range(self.kernel_size):  # kernel height
                for width in range(self.kernel_size):  # kernel width
                    area = img[height:height+dp1dc1.shape[1], width:width+dp1dc1.shape[2]]
                    tmp = np.sum(np.ravel(area, order='C') * np.ravel(dp1dc1[n], order='C'))
                    dc1dk1 = np.append(dc1dk1, tmp)
        dc1dk1 = np.reshape(dc1dk1, (self.kernel1num, 1, self.kernel_size, self.kernel_size))
        # flatten bias' gradients
        dp1dc1 = np.reshape(dp1dc1, (self.kernel1num, -1))
        dp2dc2 = np.reshape(dp2dc2, (self.kernel2num, -1))
        # update weights
        self.k1 = self.k1 - self.lr * dc1dk1
        self.kb1 = self.kb1 - self.lr * dp1dc1
        self.k2 = self.k2 - self.lr * dc2dk2
        self.kb2 = self.kb2 - self.lr * dp2dc2
        self.w1 = self.w1 - self.lr * dw1
        self.b1 = self.b1 - self.lr * db1
        self.w2 = self.w2 - self.lr * dw2
        self.b2 = self.b2 - self.lr * db2
        # return loss value(Mean Squared Error)
        return np.sum(dLda2 ** 2) / len(dLda2) * 2

    def train(self, epoch=100):
        for i in range(epoch):
            loss = 0
            indexes = np.random.permutation(len(self.train_set[1]))
            for index in tqdm(indexes):
                self.img_size = 28
                self.c1 = self.convolution(self.train_set[0][index], self.k1, self.kb1, 5)
                self.p1 = self.max_pooling(self.ReLU(self.c1))
                self.c2 = self.convolution(self.p1, self.k2, self.kb2, 10)
                self.p2 = self.max_pooling(self.ReLU(self.c2))
                self.forward(self.p2)
                loss += self.backward(index)
            print("epoch:{0}, loss:{1}, accuracy:{2}".format(i, loss / len(self.train_set[1]), self.valid()))

    def valid(self):
        acc = 0
        for i in range(len(self.valid_set[1])):
            self.img_size = 28
            self.c1 = self.convolution(self.valid_set[0][i], self.k1, self.kb1, 5)
            self.p1 = self.max_pooling(self.ReLU(self.c1))
            self.c2 = self.convolution(self.p1, self.k2, self.kb2, 10)
            self.p2 = self.max_pooling(self.ReLU(self.c2))
            self.forward(self.p2)
            if self.a2.argmax() == self.valid_set[1][i]:
                acc += 1
        return acc / len(self.valid_set[1])

    def test(self):
        acc = 0
        for i in range(len(self.test_set[1])):
            self.img_size = 28
            self.c1 = self.convolution(self.test_set[0][i], self.k1, self.kb1, 5)
            self.p1 = self.max_pooling(self.ReLU(self.c1))
            self.c2 = self.convolution(self.p1, self.k2, self.kb2, 10)
            self.p2 = self.max_pooling(self.ReLU(self.c2))
            self.forward(self.p2)
            if self.a2.argmax() == self.test_set[1][i]:
                acc += 1
        print("accuracy:{0}".format(acc / len(self.test_set[1])))

c = cnn()
c.train()
c.test()