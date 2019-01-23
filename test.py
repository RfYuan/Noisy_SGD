import numpy as np,time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt


np.random.seed(3456789)

def tanh(x):
  return np.tanh(x)
  
def d_tanh(x):
  return 1-np.tanh(x) ** 2
  
def sigmoid(x):
  return 1/ (1 + np.exp(-1*x))
  
def d_sigmoid(x):
  return sigmoid(x) * ( 1- sigmoid(x))
  
# Func: ReLu Activation Layer just for fun  
def ReLu(x):
  mask = (x>1) * 1.0
  return x * mask

def d_ReLu(x):
  mask= (x>1) * 1.0
  return  mask
  

num_epoch = 100
n_value_array = [0.01,0.3,1.0]
sample_size = 500
feature_size = 2
useful_feature_size = 2


# making 2 class that relies on 2 imformative features
# the explaination can be see :
# https://scikit-learn.org/stable/auto_examples/datasets/plot_random_dataset.html
X, Y = make_classification(n_samples=sample_size,n_features=feature_size,
                        class_sep=0.04, n_redundant=0, 
                        n_informative=useful_feature_size, n_clusters_per_class=1)

#Y = np.expand_dims(Y,axis=1)
plt.scatter(X[:,0],X[:,1],c = Y)
plt.show()

# adding an extra column so that the constant term could be trade equally
X = np.c_[X, np.ones(sample_size)]
Y = np.expand_dims(Y,axis=1)
learing_rate = 0.05

time.sleep(3)

# 0.5. Declare Hyper parameter

w1 = np.random.randn(feature_size+1,16)
w2 = np.random.randn(16,28)
w3 = np.random.randn(28,35)
w4 = np.random.randn(35,1)

for n in n_value_array:
    n_value = n

    print("Current N Value : ", n_value)


    # --------- NORMAL SGD Hyper Parameter Exact Same Weight-------------
    normal_sgd_error = 100000

    w1_normal = w1
    w2_normal = w2
    w3_normal = w3
    w4_normal = w4

    # --------- ADDITIVE SGD Hyper Parameter Exact Same Weight-------------
    ADDITIVE_sgd_error = 100000

    w1_additive = w1
    w2_additive = w2
    w3_additive = w3
    w4_additive = w4

    # --------------- Training for normal---------------
    for iter in range(num_epoch):
        layer_1 = X.dot(w1_normal)
        layer_1_act = tanh(layer_1)
        
        layer_2 = layer_1_act.dot(w2_normal)
        layer_2_act = sigmoid(layer_2)
        
        layer_3 = layer_2_act.dot(w3_normal)
        layer_3_act = tanh(layer_3)
        
        layer_4 = layer_3_act.dot(w4_normal)
        layer_4_act = sigmoid(layer_4)
        
        normal_sgd_error = np.square(layer_4_act-Y).sum()  * 0.5
        # print("Normal SGD Current iter : ",iter," Current Error: ", normal_sgd_error,end=' ')
        # a_i = activation_i (z_i)
        # z_i = W*a_(i-1)
        
        dl_da4 = layer_4_act-Y
        da4_dz4 =d_sigmoid(layer_4)
        dl_dz4 = dl_da4*da4_dz4
        dz4_dW4 = layer_3_act
        grad_4 = dz4_dW4.T.dot(dl_dz4)
        
        dl_da3 = (dl_dz4).dot(w4_normal.T)
        da3_dz3 = d_tanh(layer_3)
        dl_dz3 = dl_da3 * da3_dz3
        dz3_dW3 = layer_2_act
        grad_3 = dz3_dW3.T.dot(dl_dz3)

        dl_da2 = (dl_dz3).dot(w3_normal.T)
        da2_dz2 = d_sigmoid(layer_2)
        dl_dz2 = dl_da2 * da2_dz2
        dz2_dW2 = layer_1_act
        grad_2 = dz2_dW2.T.dot(dl_dz2)

        dl_da1 = (dl_dz2).dot(w2_normal.T)
        da1_dz1 = d_tanh(layer_1)
        dl_dz1 = dl_da1 * da1_dz1
        dz1_dW1 = X
        grad_1 = dz1_dW1.T.dot(dl_dz1)

        
        w1_normal = w1_normal - learing_rate*grad_1
        w2_normal = w2_normal - learing_rate*grad_2
        w3_normal = w3_normal - learing_rate*grad_3
        w4_normal = w4_normal - learing_rate*grad_4
        
        if (iter %100 ==0) and (iter >0):
            print("Error for Normal SGD Current iter: ",iter,"  Error: ", normal_sgd_error)

    print("Final Error for Normal SGD Current iter: ",iter,"  Error: ", np.round(normal_sgd_error,4))
    print("\n---------------------------------")

    # --------------- Training for Additive---------------
    for iter in range(num_epoch):
        layer_1 = X.dot(w1_additive)
        layer_1_act = tanh(layer_1)
        
        layer_2 = layer_1_act.dot(w2_additive)
        layer_2_act = sigmoid(layer_2)
        
        layer_3 = layer_2_act.dot(w3_additive)
        layer_3_act = tanh(layer_3)
        
        layer_4 = layer_3_act.dot(w4_additive)
        layer_4_act = sigmoid(layer_4)
        
        ADDITIVE_sgd_error = np.square(layer_4_act-Y).sum()  * 0.5
        # print("Additive Current iter : ",iter," Current Error: ", ADDITIVE_sgd_error,end='\r')
        
        dl_da4 = layer_4_act-Y
        da4_dz4 =d_sigmoid(layer_4)
        dl_dz4 = dl_da4*da4_dz4
        dz4_dW4 = layer_3_act
        grad_4 = dz4_dW4.T.dot(dl_dz4)
        
        dl_da3 = (dl_dz4).dot(w4_additive.T)
        da3_dz3 = d_tanh(layer_3)
        dl_dz3 = dl_da3 * da3_dz3
        dz3_dW3 = layer_2_act
        grad_3 = dz3_dW3.T.dot(dl_dz3)

        dl_da2 = (dl_dz3).dot(w3_additive.T)
        da2_dz2 = d_sigmoid(layer_2)
        dl_dz2 = dl_da2 * da2_dz2
        dz2_dW2 = layer_1_act
        grad_2 = dz2_dW2.T.dot(dl_dz2)

        dl_da1 = (dl_dz2).dot(w2_additive.T)
        da1_dz1 = d_tanh(layer_1)
        dl_dz1 = dl_da1 * da1_dz1
        dz1_dW1 = X
        grad_1 = dz1_dW1.T.dot(dl_dz1)
        
        # ------ Calculate The Additive weight -------
        ADDITIVE_NOISE_STD = n_value / (np.power((1 + iter), 0.55))
        ADDITIVE_GAUSSIAN_NOISE = np.random.normal(loc=0,scale=ADDITIVE_NOISE_STD)
        # ------ Calculate The Additive weight -------
        
        w1_additive = w1_additive - learing_rate*(grad_1+ ADDITIVE_GAUSSIAN_NOISE)
        w2_additive = w2_additive - learing_rate*(grad_2+ ADDITIVE_GAUSSIAN_NOISE)
        w3_additive = w3_additive - learing_rate*(grad_3+ ADDITIVE_GAUSSIAN_NOISE)
        w4_additive = w4_additive - learing_rate*(grad_4+ ADDITIVE_GAUSSIAN_NOISE)
        
        if (iter %100 ==0) and (iter >0):
            print("Error for Noisy SGD Current iter: ",iter,"  Error: ", np.round(ADDITIVE_sgd_error,4))

        
    print("Final Error for Additive SGD Current iter: ",iter,"  Error: ", np.round(ADDITIVE_sgd_error,4))
    print("\n---------------------------------\n\n")
















# -- end code --
