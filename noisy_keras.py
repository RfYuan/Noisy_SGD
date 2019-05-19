import numpy as np
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
import keras.backend as K
import matplotlib as mpl
import time
from sklearn.utils import shuffle

mpl.use('Agg')
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype(K.floatx()) / 255, axis=-1)
x_test = np.expand_dims(x_test.astype(K.floatx()) / 255, axis=-1)
y_train = np.expand_dims(y_train, axis=-1)

# np.random.shuffle(x_train)
# np.random.shuffle(y_train)
x_train,y_train = shuffle(x_train, y_train)
train_size = 10000
test_size = 5000

x_test, y_test = x_train[train_size:train_size + test_size], y_train[train_size:train_size + test_size]
x_train, y_train = x_train[:train_size], y_train[:train_size]


def indices_to_one_hot(data, nb_classes=10):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

y_train, y_test = indices_to_one_hot(y_train), indices_to_one_hot(y_test)


def custom_loss(ep, c, d, batch=1):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def noisy_loss(y_true, y_pred):
        t_shape = 10
        c_t_d = 0.5 * math.sqrt(c / ((ep) ** d))
        uniform_noise = K.random_uniform_variable(shape=(batch, t_shape), low=-c_t_d, high=c_t_d)
        uniform_noise = uniform_noise[:K.int_shape(y_true)[0]]
        mask = K.greater(K.batch_dot(uniform_noise, K.transpose(K.log(y_pred))), 0)

        mask = K.cast(mask, dtype='float32')
        new_y = y_true + uniform_noise * mask

        return K.categorical_crossentropy(new_y, y_pred)

    return noisy_loss


def get_lenet_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Conv2D(input_shape=(28, 28, 1), filters=32, kernel_size=5, activation='relu', padding='same',
                                  name='Conv-1'))
    model.add(keras.layers.MaxPool2D(pool_size=2, name='Pool-1'))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', name='Conv-2'))
    model.add(keras.layers.MaxPool2D(pool_size=2, name='Pool-2'))
    model.add(keras.layers.Flatten(name='Flatten'))
    model.add(keras.layers.Dense(units=512, activation='relu', name='Dense'))
    model.add(keras.layers.Dense(units=10, activation='softmax', name='Softmax'))
    return model


def get_model():
    model = Sequential()
    model.add(keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=3, padding='same',
                                  activation='sigmoid', use_bias=False))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax', use_bias=False))
    return model


def run_one(n, batch_=32, learning_rate=0.01, c=1, d=3):
    noisy_model, normal_model = get_model(), get_model()
    # noisy_model, normal_model = get_lenet_model(),get_lenet_model()
    normal_model.set_weights(noisy_model.get_weights())

    noisy_history = []
    history = []
    sgd = keras.optimizers.SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)
    normal_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    for r in range(n):
        loss_func = custom_loss(r + 1, c, d, batch=batch_)
        noisy_model.compile(optimizer=sgd, loss=loss_func, metrics=['accuracy'])
        noisy_model.fit(x_train, y_train, batch_size=batch_, epochs=1, verbose=0)
        normal_model.fit(x_train, y_train, batch_size=batch_, epochs=1, verbose=0)
        noisy_history.append(noisy_model.evaluate(x_test, y_test, verbose=0, batch_size=batch_))
        history.append(normal_model.evaluate(x_test, y_test, verbose=0, batch_size=batch_))
    return noisy_history, history


def run_all(avg_num, num_epoch, learning_rate, c_value, d_value, batch=1):
    noisy_his = [0] * num_epoch
    normal_his = [0] * num_epoch
    start_time = time.time()

    for k in range(avg_num):
        K.clear_session()
        n_his, his = run_one(num_epoch, learning_rate=learning_rate, c=c_value, d=d_value, batch_=batch)
        h1 = [1 - a[1] for a in his]
        h2 = [1 - a[1] for a in n_his]
        noisy_his = [a + b for a, b in zip(noisy_his, h2)]
        normal_his = [a + b for a, b in zip(normal_his, h1)]
        print("running time:", round(time.time() - start_time, 2))
    noisy_his = [a / avg_num for a in noisy_his]
    normal_his = [a / avg_num for a in normal_his]
    print(noisy_his, normal_his)
    plt.plot(range(1, num_epoch + 1), noisy_his, marker='', color='blue', label="Noisy")
    plt.plot(range(1, num_epoch + 1), normal_his, marker='', color='red', label="No Noise")
    plt.legend()
    plt.savefig(
        "10k_c=" + str(c_value) + "_d=" + str(d_value) + "_lr=" + str(learning_rate) + "batch_" + str(batch) + ".png")
    plt.clf()


c_value = 0.3
d_value = 2
num_epoch = 50
avg_num = 10
batch = 5
learning_rate = 0.01
# lenet = get_lenet_model()
# lenet.summary()
run_all(avg_num=avg_num, num_epoch=num_epoch, learning_rate=learning_rate, c_value=c_value, d_value=d_value, batch=batch)
#
# for c in [0.2]:
#     for d in range(2, 4):
#         run_all(avg_num=avg_num, num_epoch=num_epoch, learning_rate=learning_rate, c_value=c, d_value=d, batch=batch)
