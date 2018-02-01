import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from mpl_toolkits.mplot3d import Axes3D

iris = datasets.load_iris()
df_2 = pd.DataFrame(iris.data[:, :2]) #R^2 data
df_3 = pd.DataFrame(iris.data[:, :3]) #R^3 data

class kmeans(object):
    def __init__(self, data, group, k, limit, centroid, color = ["red","blue","green","orange","purple"]):
        self.data = data; self.group = group
        self.k = k; self.limit = limit
        if self.k > np.shape(data)[0]:
            print("k value is too big")
        self.dimension = np.shape(data)[1]
        self.centroid = centroid
        self.init = self.start()
        self.color = color[0:group]
        if len(self.color) != group:
            print("input more colors")
        
    def start(self):
        start = []; coord = []
        while len(start) < self.group:
            for i in range(self.dimension):
                coord = coord + [np.random.choice(self.data.ix[:,i])]            
            if coord in start:
                print("duplicate centroid")
            else:
                if len(start)>=1:
                    for t in start:
                        if self.distance(t,coord) < self.centroid:
                            break
                        start = start + [coord]
                else:
                    start = start + [coord]
            coord = []
        return(start)
    
    def distance(self,x,y):
        #euclidean distance
        if len(x) != len(y):
            return("dimension mismatch")
        else:
            dist = 0
            for i in range(len(x)):
                dist = dist + (x[i]-y[i])**2
            return(dist**(0.5))
     
    def cluster(self,init):
        distance = pd.DataFrame(np.zeros((np.shape(self.data)[0],self.group))); new = []
        for i in range(self.group):
            d = [self.distance(init[i],self.data.ix[row,:]) for row in range(np.shape(self.data)[0])]
            distance.ix[:,i] = d
        for i in range(np.shape(distance)[1]):
            neighbor = self.data.ix[np.argsort(distance.ix[:,i])[0:self.k],:]
            new = new + [np.mean(neighbor)]
        return(new)
    
    def recursive(self,init):
        out = self.cluster(init)
        if sum(np.array([self.distance(x,y) for x,y in zip(out, init)]) <= self.limit) == self.group:
            print("converged")
            return(out)
        else:
            return(self.recursive(out))
        
    def color_group(self,init):
        color_list = []; out = self.recursive(init)
        for i in range(np.shape(self.data)[0]):
            color_list = color_list + [self.color[np.argmin([self.distance(self.data.ix[i,:],y) for y in out])]]
        return([out,color_list])
    
    def two_dim_plot(self,init):
        if self.dimension != 2:
            return("dataset dimension does not match")
        out, cmap = self.color_group(init)
        plt.scatter(self.data.ix[:,0],self.data.ix[:,1],color = cmap)
        plt.scatter([out[x][0] for x in range(len(out))],[out[x][1] for x in range(len(out))],s = 80, marker = 's', c = "grey" )
        plt.title("2d data with "+str(self.group)+" clusters")
        plt.show()
        
    def three_dim_plot(self,init):
        if self.dimension != 3:
            return("dataset dimension does not match")
        out, cmap = self.color_group(init)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.data.ix[:,0],self.data.ix[:,1],self.data.ix[:,2], c=cmap)
        plt.scatter([out[x][0] for x in range(len(out))],[out[x][1] for x in range(len(out))],s = 80, marker = 's', c = "grey" )
        plt.title("3d data with "+str(self.group)+" clusters")
        plt.show()
        
            
a = kmeans(df_2,3,5,0.01,0.1)
a.two_dim_plot(a.init)

b = kmeans(df_3,3,5,0.01,0.1)
b.three_dim_plot(b.init)

