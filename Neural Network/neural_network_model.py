import numpy as np, matplotlib.pyplot as plt
import os, json, math


'''
Sources:
    1) overall structure from,
    http://neuralnetworksanddeeplearning.com/chap1.html
    2) feed forward and back propagation algorithm from,
    https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    3) SGD from,
    https://iamtrask.github.io/2015/07/27/python-network-part2/
'''   

class NN(object):
    
    def __init__(self, size, weight = None, bias = None):
        self.size = size
        if len(self.size)<3:
            return("check neural network size")
        
        if weight != None and bias != None:
            print("loading trained weights and biases")
            self.weight = weight 
            self.bias = bias
        else:
            print("producing randomly generated initial parameters")
            self.weight = [np.random.randn(row,col) for row,col in zip(self.size[1:],self.size[:-1])]
            self.bias = [np.random.randn(row,1) for row in self.size[1:]]
            print("initial parameters set")
            
               
    def sigmoid(self, numpy_array):
        if len(numpy_array)<= 0:
            return("check input array length")
        else:
            a = np.copy(numpy_array)
            return(1/(1+math.e**-a))
        
    def sigmoid_prime(self, numpy_array_passover):
        a = np.copy(numpy_array_passover)
        return(self.sigmoid(a)*(1 - self.sigmoid(a)))
            
        
    def feed_forward(self, x):
        if len(x) != self.size[0]:
            return("check input size")
        x = np.reshape(x, (len(x),1))
        '''
        dimensions of weights and biases are defined. No need to worry about those.
        unless directly defined, dimension for column vectors will be (x,), which causes error in matrix operations
        '''
        net = []; activation = [np.copy(x)]                
        for i in range(len(self.size[1:])):           
            lin_combination = np.dot(self.weight[i],activation[i]) + self.bias[i]
            sigmoid = self.sigmoid(lin_combination)
            net = net + [lin_combination]
            activation = activation + [sigmoid]
        return({"net":net,"activation":activation})
        
        
    def back_propagation(self, x, y):
        y = np.reshape(y, (len(y),1)); x = np.reshape(x, (len(x),1))
        d_w = [np.zeros(np.shape(shape_weight)) for shape_weight in self.weight]
        d_b = [np.zeros(np.shape(shape_bias)) for shape_bias in self.bias]

        model_calculation = self.feed_forward(x)
        net = model_calculation["net"]; activation = model_calculation["activation"]
        
        '''
        underlying math:
        See, https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
        let x indicate matrix multiplication
        let * indicate dot products
        using basic calculus and linear algebra,
        
        for last layer, 
            d cost / d w = (d E_tot / d out[-1]) * (d out[-1] / d net[-1]) * (d net[-1] / d w[-1])
                         = [(y_h - y) x sigmoid_prime(w*x+b)] * x
        
            let dcost = (d E_tot / d out[-1]) * (d out[-1] / d net[-1])
        
        for ith - non last layer 
            d cost / d w = dcost * (d net[-i+1] / d out[-i]) * (d out[-i] / d net[-i]) * (d net[-i] / d w[-i])
        
        Assuming cost function = Sigma(i)[(1/2)*(y_h - y)^2]
        '''
        
        d_cost = (activation[-1]-y) * self.sigmoid_prime(net[-1])
        d_b[-1] = d_cost
        d_weight = np.dot(d_cost, activation[-2].T)
        d_w[-1] = d_weight
        for i in range(2,len(self.size)):
            d_cost = np.dot(d_cost.T, self.weight[-i+1]).T * self.sigmoid_prime(net[-i])
            
            d_b[-i] = d_cost
            d_weight = np.dot(d_cost, activation[-i-1].T)
            d_w[-i] = d_weight 
        return({"dw":d_w,"db":d_b}) 
        
        
    def update_batch(self, mini, eta):
        b = [np.zeros(np.shape(bias)) for bias in self.bias]
        w = [np.zeros(np.shape(weight)) for weight in self.weight]
        for i in range(len(mini)):
            x = mini[i][0]
            y = mini[i][1]
            delta = self.back_propagation(x,y)
            b = [prev_b + change_b for prev_b, change_b in zip(b,delta["db"])]
            w = [prev_w + change_w for prev_w, change_w in zip(w,delta["dw"])]

        self.weight = [w-(eta/len(mini))*nw for w, nw in zip(self.weight, w)] 
        self.bias = [b-(eta/len(mini))*nb for b, nb in zip(self.bias, b)]           

    def sgd(self, train, epoch, batch_size, eta):
        n = len(train)
        for j in range(epoch):
            np.random.shuffle(train)
            batch = [train[k:k+batch_size] for k in range(0,n,batch_size)]
        for mini in batch:
            self.update_batch(mini, eta)
    
    def save(self, filename): 
        data = {"size": self.size,
                "weight": [w.tolist() for w in self.weight],
                "bias": [b.tolist() for b in self.bias]}
        #"cost": str(self.cost.__name__)}
        f = open(filename, "w", encoding = 'utf-8')
        json.dump(data, f)
        f.close()                
    
class json_file(object):      
    
    def load(filename):
        f = open(filename, "r")
        data = json.load(f)
        f.close()
        load_weight = [np.array(w) for w in data["weight"]]
        load_bias = [np.array(b) for b in data["bias"]]
        #cost = getattr(sys.modules[__name__], data["cost"])
        net = NN(data["size"], load_weight, load_bias)
        return(net)







