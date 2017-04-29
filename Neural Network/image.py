import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

class Read(object):
    def __init__(self, image_path, treshold, dimension):
        self.image_path = image_path
        self.word = 255
        self.blank = 0
        self.treshold = treshold
        self.dimension = dimension
        self.image = ndimage.imread(self.image_path, mode="L")
        self.original = ndimage.imread(self.image_path, mode="L")
        self.convert()
        
    def convert(self):
        for i in range(np.shape(self.image)[0]):
            for j in range(np.shape(self.image)[1]):
                if self.image[i,j] <= self.word * self.treshold:                 
                    self.image[i,j] = self.word
                else:
                    self.image[i,j] = self.blank
        
    def separate(self):
        print("reading image...")
        print("cutting image based on blank value...")
        sep_index = []; order = []
        for i in range(np.shape(self.image)[1]-1):
            if i==0:
                if sum(self.image[:,i]==self.blank) != np.shape(self.image)[0]:
                    sep_index = sep_index + [i]                         
            
            if (sum(self.image[:,i]==self.blank) == np.shape(self.image)[0] and\
                sum(self.image[:,i+1]==self.blank) != np.shape(self.image)[0]):
                sep_index = sep_index + [i]                
                                    
            elif (sum(self.image[:,i+1]==self.blank) == np.shape(self.image)[0] and\
                  sum(self.image[:,i]==self.blank) != np.shape(self.image)[0]):
                sep_index = sep_index + [i]                      
            
            if i+1 == np.shape(self.image)[1]-1:
                if sum(self.image[:,i+1]==self.blank) != np.shape(self.image)[0]:
                    sep_index = sep_index + [i+1]                           
                           
        if len(sep_index)==0:
            return("adjust treshold")
                   
        sep = []
        print(sep_index)
        for i in range(0,len(sep_index)-1,2):
            sep = sep + [self.image[:,sep_index[i]:sep_index[i+1]]]
            plt.imshow(self.image[:,sep_index[i]:sep_index[i+1]], cmap="binary")
            plt.show()
        return(sep)
    
    def resize(self):
        print("resizing image to fit neural network input layer")
        row = self.dimension[0]; col = self.dimension[1]
        images = self.separate()
        new_images = []
        turns = 1
        for i in images:
            while np.shape(i)[0] < row:
                if turns > 0:
                    i = np.concatenate([np.array([[self.blank]*np.shape(i)[1]]), i],axis=0)
                    turns = turns * -1
                else:
                    i = np.concatenate([i, np.array([[self.blank]*np.shape(i)[1]])],axis=0)
                    turns = turns * -1
                    
            while np.shape(i)[1] < col:
                if turns > 0:
                    i = np.concatenate([np.reshape(np.array([[self.blank]*np.shape(i)[0]]),(np.shape(i)[0],1)), i],axis=1)
                    turns = turns * -1
                else:
                    i = np.concatenate([i, np.reshape(np.array([[self.blank]*np.shape(i)[0]]),(np.shape(i)[0],1))],axis=1)
                    turns = turns * -1
            new_images = new_images + [i]
            plt.imshow(i, cmap="binary")
            plt.show()
        return(new_images)



