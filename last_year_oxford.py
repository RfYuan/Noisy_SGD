import numpy as np
from scipy import signal

np.random.seed(598765)


def tanh(x):
    return np.tanh(x)


def d_tanh(x):
    return 1 - np.tanh(x) ** 2


def log(x):
    return 1 / (1 + np.exp(-1 * x))


def d_log(x):
    return log(x) * (1 - log(x))

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

nb_classes = 10
def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

train_X, train_y = mnist_data()
train_X = train_X /255

x1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
x2 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
x3 = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
x4 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
X = [x1, x2, x3, x4]
Y = np.array([
    [0.53],
    [0.77],
    [0.88],
    [1.1]
])

# 0. Declare Weights
w1 = np.random.randn(2,2)*4
w2 = np.random.randn(4, 1) * 4
number_mask = w1.shape[0]


# 1. Declare hyper Parameters
num_epoch = 100
learning_rate = 0.7

cost_before_train = 0
cost_after_train = 0
final_out, start_out = np.array([[]]), np.array([[]])
# ---- Cost before training ------
for i in range(len(X)):

    layer_1 = signal.convolve2d(X[i], w1, 'valid')
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    print("***************", layer_1_act_vec.shape   )
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)
    cost = np.square(layer_2_act - Y[i]).sum() * 0.5
    cost_before_train = cost_before_train + cost
    start_out = np.append(start_out, layer_2_act)

# ----- TRAINING -------
for iter in range(num_epoch):

    for i in range(len(X)):
        layer_1 = signal.convolve2d(X[i], w1, 'valid')
        layer_1_act = tanh(layer_1)

        layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
        layer_2 = layer_1_act_vec.dot(w2)
        layer_2_act = log(layer_2)

        cost = np.square(layer_2_act - Y[i]).sum() * 0.5
        # print("Current iter : ",iter , " Current train: ",i, " Current cost: ",cost,end="\r")

        grad_2_part_1 = layer_2_act - Y[i]
        grad_2_part_2 = d_log(layer_2)
        grad_2_part_3 = layer_1_act_vec
        grad_2 = grad_2_part_3.T.dot(grad_2_part_1 * grad_2_part_2)

        grad_1_part_1 = (grad_2_part_1 * grad_2_part_2).dot(w2.T)
        grad_1_part_2 = d_tanh(layer_1)
        grad_1_part_3 = X[i]

        # print("1--",grad_1_part_1.shape, "  2--", grad_1_part_2.shape, "  3--", grad_1_part_3.shape)

        grad_1_part_1_reshape = np.reshape(grad_1_part_1, (2, 2))
        grad_1_temp_1 = grad_1_part_1_reshape * grad_1_part_2
        grad_1 = np.rot90(
            signal.convolve2d(grad_1_part_3, np.rot90(grad_1_temp_1, 2), 'valid'),
            2)

        w2 = w2 - grad_2 * learning_rate
        w1 = w1 - grad_1 * learning_rate

# ---- Cost after training ------
for i in range(len(X)):
    layer_1 = signal.convolve2d(X[i], w1, 'valid')
    layer_1_act = tanh(layer_1)

    layer_1_act_vec = np.expand_dims(np.reshape(layer_1_act, -1), axis=0)
    layer_2 = layer_1_act_vec.dot(w2)
    layer_2_act = log(layer_2)
    cost = np.square(layer_2_act - Y[i]).sum() * 0.5
    cost_after_train = cost_after_train + cost
    final_out = np.append(final_out, layer_2_act)

# ----- Print Results ---
print("\nW1 :", w1, "\n\nw2 :", w2)
print("----------------")
print("Cost before Training: ", cost_before_train)
print("Cost after Training: ", cost_after_train)
print("----------------")
print("Start Out put : ", start_out)
print("Final Out put : ", final_out)
print("Ground Truth  : ", Y.T)

# -- end code --