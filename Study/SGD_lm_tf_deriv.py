import numpy as np
import tensorflow as tf
import matplotlib as plt

#simple illustration of gradient descent in one variable linear regression
'''
SGD using derivatives
'''
x = [t for t in range(500)]
y = [(3 * j + 1) + (np.random.normal() * 20) for j in x]
rate = 0.000005
bound = 0.0001

def sgd(learning_rate, tresh, m = 0, b = 0):
    steps = 0
    while True:
        steps = steps + 1
        d_m = 0; d_b = 0
        for i in range(len(x)):            
            d_m = d_m + (-2 / len(x)) * x[i] *(y[i] - (m * x[i]) - b)
            d_b = d_b + (-2 / len(x)) * (y[i] - (m * x[i]) - b)
        m_new = m - d_m * learning_rate
        b_new = b - d_b * learning_rate
        #print([m_new, b_new])
        if abs(m_new - m) < tresh and abs(b_new - b) < tresh:
            plt.pyplot.scatter(x, y, color = "r")
            plt.pyplot.plot(x, [m_new * z + b_new for z in x], color="k")
            return([m_new, b_new, steps])
        else:
            m = m_new
            b = b_new

sgd(rate, bound)
#slope converges to 3 

'''
SGD using tensorflow
'''

x_data = [t for t in range(500)]
y_data = [(3 * j + 1) + (np.random.normal() * 20) for j in range(500)]
tf_rate = 0.0000005 

def sgd_tf(rate, total_epoch = 100):   
    #placeholder
    x = tf.placeholder('float')
    y = tf.placeholder('float')
    #parameters to be optimized
    w = tf.Variable(np.random.randn(), name = "m") 
    b = tf.Variable(np.random.randn(), name = "b") 
    y_hat = (x * w) + b
    cost = tf.square(y_hat - y)
    #Variable objects are trained by default                  
    gd = tf.train.GradientDescentOptimizer(rate).minimize(cost)  
    
    with tf.Session() as sess:        
        sess.run(tf.global_variables_initializer())        
        for epoch in range(total_epoch):
            loss = 0
            for x_train, y_train in zip(x_data, y_data):           
                sess.run(gd, feed_dict = {x:x_train, y:y_train})
                loss = loss + sess.run(cost, feed_dict = {x:x_train, y:y_train})
            print("epoch",epoch, "loss:", loss)
        w_update, b_update = sess.run([w, b])
        print(w_update)
        
           
sgd_tf(tf_rate)
#Amount of loss dramatically decreases after first epoch. w converges to 3. 
