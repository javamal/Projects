import os
import json
import numpy as np
import matplotlib.pyplot as plt
import digit_data
from nn import NN, json_file #import class from python file
from image import Read

os.chdir = 'C:\\Users\\donkey\\Desktop\\machine learning\\Neural Network'

'''
1. Import neural network model, with trained parameters
neural network - train weights and bias - dump parameters in json file
'''
model = NN([784, 30, 10]) #untrained
train, val, test = digit_data.load_data_wrapper()
model.sgd(train, epoch = 50, batch_size = 10, eta = 2)   
model.save("neural_network_digit.json") #with 10 reps of training

trained = json_file.load("neural_network_digit.json")  #load back trained parameters



'''
2. testing data from NMIST
'''
         
def test_model(model,i):
    t = test[i] #ith tuple
    out = model.feed_forward(t[0])["activation"][-1] #last output layer
    print("actual output: "+ str(t[1]))
    print(out)
    print("model prediction: "+str(np.where(out == max(out))[0][0] ))
    if t[1] == np.where(out == max(out))[0][0]:
        return(True)
    else:
        return(False)
        
def predict(model, rep): #variable model takes in neural network instance model
    success = 0
    for i in range(rep):
        if test_model(model,i) == True:
            success = success + 1
    print("prediction rate: "+str(float(success/rep)))
    return(float(success/rep))
    
predict(trained, len(test))
       

'''
#trained model vs un-trained model?

predict(trained, 100)
predict(model, 100)

Un-trained model (randomized initial parameters) serves no predicting function
with prediction rate of 10%, which would be the rate I would get by randomly choosing between 0 and 9.

Trained model (trained weights and parameters from saved json file) 
predicts 97% of test cases correctly.
'''

















