import math
import random
import numpy as np

class Node(object):
    def __init__(self,depth,n,p,value=0):
        self.depth=depth
        self.n=n
        self.p=p
        self.value=value
        self.subtree=[]
        self.create_subtree()
        
    def create_subtree(self):
        if self.depth>0:
            for i in [1,2]:
                remain=self.n-i
                if remain>=0:
                    self.subtree.append( Node(self.depth-1,remain,-self.p, self.assign(remain,-self.p)) )
                
    def assign(self,remain,p):
        if remain==0:
            if p==1:
                return(math.inf)
            else:
                return(-math.inf)
        else:
            return(0)

def reverse_deduction(node,player,alpha=-math.inf,beta=math.inf): #game theory - dynamic game
    if len(node.subtree)==0:
        return(node.value)
    
    if player==1:
        init=-math.inf
        for i in node.subtree:
            init=max(init,reverse_deduction(i,-player))
            alpha=max(alpha,init)
            if alpha>beta:
                break
        return(init)
    else:
        init=math.inf
        for i in node.subtree:
            init=min(init,reverse_deduction(i,-player))
            beta=min(beta,init)
            if alpha>beta:
                break
        return(init)

def check(n,p):
    if n==0:
        if p==1: #call function before changing player
            print("computer wins")
            return(True)
        else:
            print("human wins")
            return(True)
    else:
        return(False)
            
def play(n):
    initial=n
    p=1
    while True:
        while True:
            move=input("1 or 2: ")
            if move!="1" and move!="2":
                print("wrong input value")
            elif n-int(float(move)) < 0:
                print("only "+str(n)+" stick(s) remaining")
            else:
                break
        n=n-int(float(move))     
        print("human plays "+str(move)+". Remaining: "+str(n))
        if check(n,p)==False: #Computer plays
            p=-p #switch
            choice=Node(n,n,p) #create node where comp plays
            options=[]
            if len(choice.subtree)>0: #will ot touch terminal nodes
                for i in choice.subtree:
                    options=options+[reverse_deduction(i,-p)] #subtree starts with oppontent
            print("Current state: Terminal node score for computer\n"+str(options))
                        
            if len(options)==2:
                if options[0] == options[1]:
                    print("computer plays randomly generated move")
                    computer_move = [1,2][random.randint(0,1)]
                else:
                    computer_move = [1,2][np.where(np.array(options)==min(options))[0][0]]
            elif len(options)==1:
                computer_move = [1,2][0]
                                                              
            n=n-computer_move
            if n<0:
                return("only "+str(n)+" remaining")
            print("computer plays "+str(computer_move)+". Remaining: "+str(n))
            if check(n,p)==True:
                play_again=input("play again (y/n)? ")
                if play_again=="y":
                    play(initial)
                elif play_again=="n":
                    return("end game")
            else:
                p=(-1)*p
        else:
            play_again=input("play again (y/n)? ")
            if play_again=="y":
                play(initial)
            elif play_again=="n":
                return("end game")

play(15)
