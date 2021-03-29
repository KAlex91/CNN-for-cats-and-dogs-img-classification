
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time
import os


X=pickle.load(open("X.pickle","rb"))
y=pickle.load(open("y.pickle","rb"))

X=np.array(X/255.0)
y=np.array(y)

## in my logs i tried many different parameters for the model. next to each parameter i will comment what else i tested in this project


dense_layers = [1]  # tested [1,2]
layer_sizes = [128] # tested [64,128]
conv_layers = [3]  #tested [2,3]
dense_sizes = [256] #tested [128.256]
conv_dims = [3]  #tested [3,4]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            for dense_size in dense_sizes:
                for conv_dim in conv_dims:
                    NAME = "{}_conv_{}x{}_{}_nodes_{}_dense_{}_d-nodes{}".format(conv_layer, conv_dim, conv_dim ,layer_size, dense_layer, dense_size, int(time.time()))
                    tboard_log_dir = os.path.join("logs",NAME)
                    tensorboard = TensorBoard(log_dir = tboard_log_dir)    
                    
                    model = Sequential()
                    
                    model.add( Conv2D(layer_size, (conv_dim,conv_dim), input_shape = X.shape[1:]) )
                    model.add(Activation("relu"))
                    model.add(MaxPooling2D(pool_size = (2,2)))
                    
                    for l in range(conv_layer -1):
                        model.add(Conv2D(layer_size, (conv_dim,conv_dim)) )
                        model.add(Activation("relu"))
                        model.add(MaxPooling2D(pool_size = (2,2)))
                        
                    model.add(Flatten())
                    
                    for l in range(dense_layer):
                        model.add(Dense(dense_size))
                        model.add(Activation("relu"))
                        model.add(Dropout(0.2))
                 
                    model.add(Dense(1))
                    model.add(Activation("sigmoid"))
                    
                    model.compile(loss = 'binary_crossentropy' ,
                              optimizer = 'adam',
                              metrics = ['accuracy']   )
                    
                    model.fit(X, y, batch_size = 32 , epochs = 5, validation_split = 0.2 , callbacks = [tensorboard] )
        

model.save('128x3_CNN.model')        
## so far best model was 3_conv_3x3_128_nodes_1_dense_256_dnodes with: loss:0.2932  accuracy:0.8735   val_loss:0.3659  val_acc:0.8405
## in cmd ---> tensorboard --logdir=logs/


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            