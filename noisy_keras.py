import numpy as np
import keras
import math
from keras.datasets import mnist
from keras.models import Sequential
import keras.backend as K
import matplotlib as mpl
import time

mpl.use('Agg')
import sys
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype(K.floatx()) / 255, axis=-1)
x_test = np.expand_dims(x_test.astype(K.floatx()) / 255, axis=-1)

y_train, y_test = np.expand_dims(y_train, axis=-1), np.expand_dims(y_test, axis=-1)

x_test = x_train[1000:2000]
y_test = y_train[1000:2000]
x_train = x_train[:1000]
y_train = y_train[:1000]
print(y_train[100])
def indices_to_one_hot(data, nb_classes=10):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

y_train,y_test = indices_to_one_hot(y_train),indices_to_one_hot(y_test)

def custom_loss(ep,c,d, batch=1):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def noisy_loss(y_true, y_pred):
        t_shape = 10
        c_t_d = 0.5 * math.sqrt(c / ( (ep) ** d))
        uniform_noise = K.random_uniform_variable(shape= (batch,t_shape), low = -c_t_d, high= c_t_d)
        mask = K.greater( K.batch_dot(uniform_noise, K.transpose(K.log(y_pred))),0)
        
        mask = K.cast(mask,dtype= 'float32')
        new_y = y_true + uniform_noise*mask
        
        return K.categorical_crossentropy(new_y, y_pred)
    return noisy_loss

def get_model():
    model = Sequential()
    model.add(keras.layers.Conv2D(input_shape=(28, 28, 1), kernel_size=(3,3), filters=3, padding='same',activation='sigmoid'))
    model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model

def run_all(n,batch_=1):
    noisy_model = get_model()
    normal_model = get_model()
    noisy_history = []
    history=[]
    sgd= keras.optimizers.SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)
    normal_model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
    for r in range(n):
        loss_func = custom_loss(r+1,1,2,batch =batch_)
        noisy_model.compile(optimizer=sgd,loss=loss_func,metrics=['accuracy'])
        noisy_model.fit(x_train,y_train, batch_size=1,epochs=1,verbose=0)
        normal_model.fit(x_train,y_train, batch_size=1,epochs=1,verbose=0)
        noisy_history.append(noisy_model.evaluate(x_test, y_test))
        history.append(normal_model.evaluate(x_test, y_test))
    return noisy_history,history
num_epoch = 50
noisy_his = [0]*num_epoch
normal_his = [0]*num_epoch
start_time = time.time()

for k in range(1):
    n_his,his = run_all(num_epoch)
    h1 = [1-a[1] for a in his]
    h2 = [1-a[1] for a in n_his]
    noisy_his = [a+b for a,b in zip(noisy_his,h2)]
    normal_his = [a+b for a,b in zip(normal_his,h1)]
    print("running time:", round(time.time() - start_time, 2))

plt.plot(range(1, num_epoch + 1), noisy_his, marker='', color='blue', label="Noisy")
plt.plot(range(1, num_epoch + 1), normal_his, marker='', color='red', label="No Noise")
plt.legend()
plt.savefig("1.png")
plt.clf()
   
print(his)
print(n_his)