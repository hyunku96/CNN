import pickle, gzip
import numpy as np
#parameter's type of np.reshape is tuple!!
class cnn:
    def __init__(self, kernel_size=3, kernel_num=10, lr=0.01, img_size=28):
        # Load the dataset -> tuple, [0][index]:data, [1][index]:label
        with gzip.open('mnist.pkl.gz', 'rb') as f:
            self.train_set, self.valid_set, self.test_set = pickle.load(f, encoding='latin1')
        self.img_size = img_size
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.result = [] #store for backprop
        #cnn weights
        self.k1 = np.random.rand(kernel_size * kernel_size, kernel_num) - 0.5
        self.kb1 = np.random.rand(img_size * img_size, kernel_num) - 0.5
        #fc weights
        self.w1 = np.random.rand(14 * 14 * 10, 70) - 0.5 #input * hidden
        self.b1 = np.random.rand(70) - 0.5
        self.w2 = np.random.rand(70, 10) - 0.5 #hidden * output
        self.b2 = np.random.rand(10) - 0.5
        #fc outputs
        self.o1 = None
        self.a1 = None
        self.o2 = None
        self.a2 = None
        #max pooling output
        self.p = []
        #learning rate
        self.lr = lr

    def convolution(self, img):
        self.result = []
        #reshape & same padding to img
        padding = int(self.kernel_size / 2)
        self.img_size = 28 #don't fix
        img = img.reshape(self.img_size, self.img_size) #don't fix shape of img
        img = np.pad(img, ((padding, padding), (padding, padding)), 'constant', constant_values=(0))

        for a in range(self.kernel_num):
            out = []
            for i in range(padding, img.shape[0] - padding):
                for j in range(padding, img.shape[1] - padding):
                    '''
                    tmp = 0
                    for k in range(-padding, padding + 1):
                        for l in range(-padding, padding + 1):
                            tmp += img[i + k][j + l] * self.k1[k * self.kernel_size + l, a]
                    '''
                    area = img[i-padding:i+padding+1][j-padding:j+padding+1]
                    tmp = np.sum(np.ravel(area, order='C') * self.k1[:, a])

                    out.append(tmp)
            out = out + self.kb1[:, a]
            self.result.append(np.reshape(out, (self.img_size, self.img_size))) #28, 28
        #result's shape 10, 28, 28

    def ReLU(self, x):
        return np.maximum(0, x)

    def max_pooling(self, img, size=2):
        self.p = []
        for a in range(self.kernel_num):
            out = []
            for i in range(0, img.shape[1], size):
                for j in range(0, img.shape[2], size):
                    '''
                    pool = []
                    for k in range(size):
                        for l in range(size):
                            pool.append(img[i + k, j + l, a]) #img(convolution) shape : (28, 28, 10)
                    out.append(max(pool))
                    '''
                    out.append(np.max(img[a, i:i+size, j:j+size]))

            out = np.array(out)
            self.img_size = 14 #don't fix
            self.p.append(np.reshape(out, (self.img_size, self.img_size)))
        #p's shape 10, 14, 14

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, img):
        img = np.ravel(img, order='C')
        self.o1 = np.dot(img, self.w1) + self.b1
        self.a1 = self.sigmoid(self.o1)
        self.o2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(self.o2)

    def backward(self, index):
        # one-hot
        label = np.zeros(10)
        label[self.train_set[1][index]] = 1
        #back propagation with partial differentiation
        dLda2 = self.a2 - label #10, 1
        da2do2 = self.sigmoid(self.o2) * (1 - self.sigmoid(self.o2)) #10, 1
        da1do1 = self.sigmoid(self.o1) * (1 - self.sigmoid(self.o1)) #70, 1
        #calculate gradients
        db2 = dLda2 * da2do2 #10, 1
        dw2 = np.dot(self.a1.reshape(-1, 1), db2.reshape(1, -1)) #70, 10
        db1 = np.dot(self.w2, db2) * da1do1 #70, 1
        dw1 = np.dot(np.reshape(self.p, (self.img_size * self.img_size * self.kernel_num, 1)), db1.reshape(1, -1)) #1960, 90
        #ReLU & maxpooling layer's gradient
        grad = np.dot(self.w1, db1)
        grad = grad.reshape(self.img_size, self.img_size, self.kernel_num) #14, 14, 10
        self.img_size = 28 #don't fix
        do1dpr = np.zeros((self.img_size, self.img_size, self.kernel_num)) #28, 28, 10
        #make gradient map
        for a in range(self.kernel_num):
            for i in range(0, self.img_size, 2):
                for j in range(0, self.img_size, 2):
                    area = self.result[a][i:i+2, j:j+2]
                    if area.any() > 0: #if their exists positive values
                        maxloc = area.argmax()
                        do1dpr[i + int(maxloc / 2), j + maxloc % 2, a] = grad[int(i/2), int(j/2), a]
        #convolution layer's gradient
        padding = int(self.kernel_size / 2)
        img = self.train_set[0][index].reshape(28, 28) #don't fix
        img = np.pad(img, ((padding, padding), (padding, padding)), 'constant', constant_values=(0))
        #calculate kernel's gradient
        dprdk1 = np.array([])
        for a in range(self.kernel_num):
            for i in range(padding * 2 + 1):
                for j in range(padding * 2 + 1):
                    tmp = 0
                    for k in range(do1dpr.shape[0]):
                        for l in range(do1dpr.shape[1]):
                            tmp += img[i + k][j + l] * do1dpr[k, l, a]
                    dprdk1 = np.append(dprdk1, tmp)
        dprdk1 = np.reshape(dprdk1, (-1, self.kernel_num))
        do1dpr = np.reshape(do1dpr, (-1, self.kernel_num))
        #do1dpr = np.ravel(do1dpr, order='C')
        #update weights
        self.k1 = self.k1 - self.lr * dprdk1
        self.kb1 = self.kb1 - self.lr * do1dpr
        self.w1 = self.w1 - self.lr * dw1
        self.b1 = self.b1 - self.lr * db1
        self.w2 = self.w2 - self.lr * dw2
        self.b2 = self.b2 - self.lr * db2
        #return loss value(Mean Squared Error)
        return np.sum(dLda2**2) / len(dLda2) * 2

    def train(self, epoch=10):
        for i in range(epoch):
            loss = 0
            indexes = np.random.permutation(len(self.train_set[1]))
            for index in indexes:
                self.convolution(self.train_set[0][index])
                self.max_pooling(self.ReLU(self.result))
                self.forward(self.p)
                loss += self.backward(index)
            print("epoch:{0}, loss:{1}, accuracy:{2}".format(i, loss / len(self.train_set[1]), self.valid()))

    def valid(self):
        acc = 0
        for i in range(len(self.valid_set[1])):
            self.convolution(self.valid_set[0][i])
            self.max_pooling(self.ReLU(self.result))
            self.forward(self.p)
            if self.a2.argmax() == self.valid_set[1][i]:
               acc += 1
        return acc / len(self.valid_set[1])

    def test(self):
        acc = 0
        for i in range(len(self.test_set[1])):
            self.convolution(self.test_set[0][i])
            self.max_pooling(self.ReLU(self.result))
            self.forward(self.p)
            if self.a2.argmax() == self.test_set[1][i]:
               acc += 1
        print("accuracy:{0}".format(acc / len(self.test_set[1])))

c = cnn()
c.train()
c.test()