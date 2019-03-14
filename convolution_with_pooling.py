import numpy as np
import time
import math
from scipy import signal
import skimage.measure
import matplotlib as mpl

mpl.use('Agg')
import sys
import matplotlib.pyplot as plt


# np.random.seed(13579)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLu(x):
    return np.maximum(x, 0)


def d_ReLu(x):
    return (x > 0) * 1.0


def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def d_softmax(softmax, x):
    jacobian = softmax_grad(softmax)
    return jacobian.dot(x.T)

def cross_entropy(labels, prediction ):
    return -np.sum( labels * np.log(prediction))

def d_cross_entropy(labels, prediction):
    return -labels/prediction

def loadMNIST(prefix, folder):
    intType = np.dtype('int32').newbyteorder('>')
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile(folder + "/" + prefix + '-images-idx3-ubyte', dtype='ubyte')
    magicBytes, nImages, width, height = np.frombuffer(data[:nMetaDataBytes].tobytes(), intType)
    data = data[nMetaDataBytes:].astype(dtype='float32').reshape([nImages, width, height])

    labels = np.fromfile(folder + "/" + prefix + '-labels-idx1-ubyte',
                         dtype='ubyte')[2 * intType.itemsize:]
    return data, labels


def mnist_data():
    trainingImages, trainingLabels = loadMNIST("train", ".")
    # testImages, testLabels = loadMNIST("t10k", ".")
    return trainingImages, trainingLabels


def indices_to_one_hot(data, nb_classes=10):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


train_X, train_y = mnist_data()
train_X = train_X / 255
train_y = indices_to_one_hot(train_y)
train_size = 1000
train_X = train_X[:train_size]
train_y = train_y[:train_size]
plt.ioff()

# 0. Declare Weights
w1 = np.random.randn(3, 3, 3) * 3
w2 = np.random.randn(14 * 14 * 3, 10)
number_mask = w1.shape[0]
layer_1 = np.zeros((28, 28, 3))

# 1. Declare hyper Parameters
num_epoch = 100
learning_rate = float(sys.argv[1])
ind1 = int(sys.argv[2])
ind2 = int(sys.argv[3])
noise_types = {'G': 'Gaussian', 'U': 'Uniform', 'None': 'None'}
noise = 'None'
try:
    noise = noise_types[str(sys.argv[4])]
except Exception as e:
    noise = 'None'
c_value = [0.2 * i for i in range(16)]
d_value = list(range(1, 6))


# cost_before_train = 0
# cost_after_train = 0


def train_set_error_rate(w1_=w1, w2_=w2, train_X_=train_X, train_y_=train_y, layer_1=layer_1):
    count_error = 0
    # ---- Cost after training ------
    for i in range(len(train_X_)):
        for j in range(number_mask):
            layer_1[:, :, j] = signal.convolve2d(train_X[i], w1_[:, :, j], mode='same')
        layer_1_act = sigmoid(layer_1)

        layer_1_pooled = skimage.measure.block_reduce(layer_1_act, (2, 2, 1), func=np.max)
        layer_1_pooled_vec = np.expand_dims(np.reshape(layer_1_pooled, -1), axis=0)

        layer_2 = layer_1_pooled_vec.dot(w2_)
        layer_2_act = softmax(layer_2)
        # if i % 100 == 0:
        # print("different of the ", i, "th training sample:", np.round(layer_2_act - train_y[i], decimals=2), "   ")
        count_error += np.argmax(layer_2_act) != np.argmax(train_y_[i])
    error_rate = count_error / len(train_y_)
    return error_rate


def train_convolutional_network(train_X=train_X, train_y=train_y.copy(), noise_='None',
                                learning_rate_=learning_rate,
                                num_epoch_=num_epoch, w1_=w1, w2_=w2, layer_1=layer_1,
                                c=c_value[ind1], d=d_value[ind2]):
    # ----- TRAINING -------
    error_list = []
    train_size_ = len(train_X)
    for r in range(num_epoch_):
        if r % 700 == 0:
            learning_rate_ = learning_rate_ * 0.5
        # print("------------iteration:", iter, "-----------")
        for i in range(train_size_):
            for j in range(number_mask):
                layer_1[:, :, j] = signal.convolve2d(train_X[i], w1_[:, :, j], mode='same')
            layer_1_act = sigmoid(layer_1)

            layer_1_pooled = skimage.measure.block_reduce(layer_1_act, (2, 2, 1), func=np.max)
            layer_1_pooled_vec = np.expand_dims(np.reshape(layer_1_pooled, -1), axis=0)

            layer_2 = layer_1_pooled_vec.dot(w2_)
            layer_2_act = softmax(layer_2)

            # adding noise
            new_y = train_y[i]
            if noise_ == "Uniform":
                c_t_d = 0.5 * math.sqrt(c / ((i + 1 + train_size_ * r) ** d))
                uniform_noise = np.random.uniform(-c_t_d, c_t_d, (1, 10))
                if uniform_noise.dot(np.log(layer_2_act).T) > 0:
                    new_y = train_y[i] + uniform_noise
                    # train_y[i] = train_y[i] + uniform_noise
            elif noise == "Gaussian":
                variance = math.sqrt(c / ((i + 1 + train_size_ * r) ** d * 12))
                gaussian_noise = np.random.normal(0, variance, (1, 10))
                if gaussian_noise.dot(np.log(layer_2_act).T) > 0:
                    new_y = train_y[i] + gaussian_noise
                    # train_y[i] = train_y[i] + uniform_noise

            # grad_2_part_1 = layer_2_act - new_y
            grad_2_part_1 = d_cross_entropy(new_y,layer_2_act)
            # grad_2_part_1 = layer_2_act - train_y[i]
            grad_2_part_2 = softmax_grad(layer_2_act)
            grad_2_part_3 = layer_1_pooled_vec
            grad_2 = grad_2_part_3.T.dot(grad_2_part_1.dot(grad_2_part_2))

            # differentiation from before
            grad_1_part_1_temp = np.reshape((grad_2_part_1.dot(grad_2_part_2)).dot(w2_.T), (14, 14, 3))
            grad_1_window = grad_1_part_1_temp.repeat(2, axis=0).repeat(2, axis=1)  # expand dimension for pool
            # grad_1_mask = np.full((28,28,3), 0.25, dtype=float)
            grad_1_mask = np.equal(layer_1_act, layer_1_pooled.repeat(2, axis=0).repeat(2, axis=1)).astype(int)
            grad_1_part_1 = grad_1_window * grad_1_mask  # shape = (28,28,3)
            grad_1_part_2 = d_sigmoid(layer_1)

            grad_1_part_3 = np.zeros((30, 30))
            grad_1_part_3[1:29, 1:29] = train_X[i]

            # print("1--",grad_1_part_1.shape, "  2--", grad_1_part_2.shape, "  3--", grad_1_part_3.shape)

            temp = grad_1_part_1 * grad_1_part_2
            # print("shape_temp-------", temp.shape, "grad_1_part_3----------", grad_1_part_3.shape )
            grad_1 = np.zeros((3, 3, 3))

            for j in range(number_mask):
                grad_1[:, :, j] = np.rot90(
                    signal.convolve2d(grad_1_part_3, np.rot90(temp[:, :, j], 2), mode='valid'),
                    2)
            # if (i % 100 == 0)and(iter %10 ==0)and(iter>0):
            #     print("grad1  norm:",np.sum(grad_1), "       grad2 norm", np.sum(grad_2))
            w2_ = w2_ - grad_2 * learning_rate_
            w1_ = w1_ - grad_1 * learning_rate_
        error_list.append(train_set_error_rate(w1_=w1_, w2_=w2_))

    return w1_, w2_, error_list


noisy_error_history = [0] * num_epoch
noiseless_error_history = [0] * num_epoch

for k in range(15):
    print("start training--------------")
    start_time = time.time()

    for i in range(20):
        w1_without_noise, w2_without_noise, error_noiseless = train_convolutional_network(noise_="None")
        w1, w2, error_noisy = train_convolutional_network(noise_=noise)
        noisy_error_history = [a + b for a, b in zip(noisy_error_history, error_noisy)]
        noiseless_error_history = [a + b for a, b in zip(noiseless_error_history, error_noiseless)]
    noisy_error_history = [a / 20 for a in noisy_error_history]
    noiseless_error_history = [a / 20 for a in noiseless_error_history]

    print("running time:", round(time.time() - start_time, 2))

    plt.plot(range(1, num_epoch + 1), noisy_error_history, marker='', color='blue', label="Noisy")
    plt.plot(range(1, num_epoch + 1), noiseless_error_history, marker='', color='red', label="No Noise")
    plt.legend()
    plt.savefig(
        "epoch100_c=" + str(c_value[ind1]) + "__d=" + str(d_value[ind2]) + "_" + "Noise=" + (sys.argv[4]) + '__' + str(
            k) + ".png")
    plt.clf()
    # plt.show()
