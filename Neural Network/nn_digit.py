import os, json, numpy as np, matplotlib.pyplot as plt
os.chdir = 'C:\\Users\\donkey\\Desktop\\machine learning\\Neural Network'
from nn import NN, json_file #import class from python file
from image import Read
import digit_data


'''
1. Import neural network model, with trained parameters
neural network - train weights and bias - dump parameters in json file
'''
model = NN([784, 30, 10])
train,val,test = digit_data.load_data_wrapper()
model.sgd(train, epoch = 50, batch_size = 10, eta = 2)   
model.save("neural_network_digit.json") #with 10 reps of training

trained = json_file.load("neural_network_digit.json")  #load back trained parameters



'''
2. testing text from cropped iamges
'''
a = Read("C:\\Users\\donkey\\Desktop\\machine learning\\Neural Network\\num.jpg",0.9,[28,28])
plt.imshow(a.image, cmap="binary")
b = a.resize()

def image_predict(i):
    b = np.reshape(i,(784,1))
    output = trained.feed_forward(b)["activation"][-1]
    return(np.where(output == max(output))[0][0])

image_predict(b[4])



'''
3. testing testing data from NMIST
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

'model' (not trained model randomized initial parameters) serves no predicting function
with prediction rate of 10%, which would be the rate I would get by randomly choosing between 0 and 9.

The trained model ( which takes trained weights and parameters from saved json file) 
predicts 97% of test cases correctly.
'''


'''
questions
1) how does SGA work? it seems too simple,,, -> resolved (gradient descent)
2) how does update batch work? why do we add partial derivative? -> resolved (gradient is sum of partial derivatives)
3) how do we decide how much hidden layers we want to have? Further, how do we decide how much neuron -> don't know
should we assign to each hidden layers?
'''  


















