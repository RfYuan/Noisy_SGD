import numpy as np
from scipy import signal

# np.random.seed(598765)


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
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)
def d_softmax(softmax, x):
    jacobian = softmax_grad(softmax)
    return jacobian.dot(x.T)

def loadMNIST( prefix, folder ):
    intType = np.dtype( 'int32' ).newbyteorder( '>' )
    nMetaDataBytes = 4 * intType.itemsize

    data = np.fromfile( folder + "/" + prefix + '-images.idx3-ubyte', dtype = 'ubyte' )
    magicBytes, nImages, width, height = np.frombuffer( data[:nMetaDataBytes].tobytes(), intType )
    data = data[nMetaDataBytes:].astype( dtype = 'float32' ).reshape( [ nImages, width, height ] )

    labels = np.fromfile( folder + "/" + prefix + '-labels.idx1-ubyte',
                          dtype = 'ubyte' )[2 * intType.itemsize:]
    return data, labels

def mnist_data():
    trainingImages, trainingLabels = loadMNIST( "train", "." )
    testImages, testLabels = loadMNIST( "t10k", "." )
    return trainingImages, trainingLabels

def indices_to_one_hot(data, nb_classes=10):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

train_X, train_y = mnist_data()
train_X = train_X /255
train_y = indices_to_one_hot(train_y)
train_X = train_X[:1000]
train_y = train_y[:1000]

# 0. Declare Weights
w1 = np.random.randn(3, 3, 3)*3
w2 = np.random.randn(14*14*3,10)
number_mask = w1.shape[0]


# 1. Declare hyper Parameters
num_epoch = 1000
learning_rate = 0.2
c_value = [0.2*i for i in range(15)]
cost_before_train = 0
cost_after_train = 0
# final_out, start_out = np.array([[]]), np.array([[]])
layer_1 = np.zeros((14, 14, 3))
# ---- Cost before training ------
for i in range(len(train_X)):
    for j in range(number_mask):
        layer_1[:, :, j] = signal.convolve2d(train_X[i], w1[:, :, j], mode='same')[::2,::2]
    layer_1_act = sigmoid(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)

    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = softmax(layer_2)
    cost = np.square(layer_2_act - train_y[i]).sum() * 0.5
    cost_before_train = cost_before_train + cost
    # start_out = np.append(start_out, layer_2_act)

# ----- TRAINING -------
for iter in range(num_epoch):
    if iter%300==0:
        learning_rate = learning_rate *0.5
    # print("------------iteration:", iter, "-----------")
    for i in range(len(train_X)):
        for j in range(number_mask):
            layer_1[:, :, j] = signal.convolve2d(train_X[i], w1[:, :, j], mode='same')[::2, ::2]
        layer_1_act = sigmoid(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)

        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = softmax(layer_2)

        cost = np.square(layer_2_act - train_y[i]).sum() * 0.5

        # if i == 999:
        #     print("Current iter : ",iter , " Current train sample: ",i, " Current cost: ",cost )

        grad_2_part_1 = layer_2_act - train_y[i]
        # grad_2_part_2 = d_sigmoid(layer_2)
        grad_2_part_2 = softmax_grad(layer_2_act)
        grad_2_part_3 = layer_1_act_vec
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1.dot(grad_2_part_2) )

        grad_1_part_1 = (grad_2_part_1.dot(grad_2_part_2)).dot(w2.T)
        grad_1_part_2 = d_tanh(layer_1)
        grad_1_part_3 = train_X[i]

        # print("1--",grad_1_part_1.shape, "  2--", grad_1_part_2.shape, "  3--", grad_1_part_3.shape)

        grad_1_part_1_reshape = np.reshape(grad_1_part_1, (14,14,3))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        # print(grad_1_temp_1.shape)
        grad_1 = np.zeros((3,3,3))

        grad_1_part_3_t = np.zeros((30,30))
        grad_1_part_3_t[1:29,1:29] = grad_1_part_3


        for j in range(number_mask):
            temp = np.zeros((28,28))
            temp[1::2,1::2] = grad_1_temp_1[:,:,j]
            grad_1[:, :, j] = np.rot90(
                signal.convolve2d(grad_1_part_3_t, np.rot90(temp, 2), 'valid'),
                2)
        # if (i % 100 == 0)and(iter %10 ==0)and(iter>0):
        #     print("grad1  norm:",np.sum(grad_1), "       grad2 norm", np.sum(grad_2))
        w2 = w2 - grad_2 * learning_rate
        w1 = w1 - grad_1 * learning_rate

# ---- Cost after training ------
for i in range(len(train_X)):
    for j in range(number_mask):
        layer_1[:, :, j] = signal.convolve2d(train_X[i], w1[:, :, j], mode='same')[::2,::2]
    layer_1_act = sigmoid(layer_1)
    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)

    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = softmax(layer_2)
    if i %100 ==0 :
        print("different of the " ,i, "th training sample:" , np.round(layer_2_act-train_y[i], decimals=2), "   ")
    cost = np.square(layer_2_act - train_y[i]).sum() * 0.5
    cost_after_train = cost_after_train + cost
    # final_out = np.append(final_out, layer_2_act)

# ----- Print Results ---
# print("\nW1 :", w1, "\n\nw2 :", w2)
print("----------------")
print("Cost before Training: ", cost_before_train)
print("Cost after Training: ", cost_after_train)
print("----------------")
# print("Start Out put : ", start_out)
# print("Final Out put : ", final_out)
# print("Ground Truth  : ", train_y.T)

# -- end code --