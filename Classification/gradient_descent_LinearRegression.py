import numpy as np
import matplotlib as plt

#simple illustration of gradient descent in one variable linear regression
#same idea is used for gradient descent in neural networks

x = [t for t in range(100)]
y = [(3 * j + 1) + (np.random.normal() * 10) for j in x]

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
            
sgd(0.000005, 0.0001)
