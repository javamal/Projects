import sys, math,random
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
    '''
    did not assign depth as variable.
    will search until no subtrees exist
    '''
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
'''
strategy

reverse_deduction(Node(4,4,1,0),1) #start with n=4, first to play loses
reverse_deduction(Node(3,3,1,0),1) #start with n=3, first to play wins
reverse_deduction(Node(5,5,1,0),1) #start with n=5, first to play gets to n=3 and wins   
reverse_deduction(Node(1,1,1,0),1)  
reverse_deduction(Node(0,1,1,0),1) #terminal node
'''

def check(n,p):
    if n==0:
        if p==1: #call function before changing player
            print("computer")
            return(True)
        else:
            print("human")
            return(True)
    else:
        return(False)
            
def play(n):
    initial=n
    p=1
    while True:
        move=input("1 or 2: ")
        if move!="1" and move!="2":
            return("wrong input value")
        n=n-int(float(move))
        if n<0:
            return("only "+str(n)+" remaining")
        print("human plays "+str(move)+". Remaining: "+str(n))
        if check(n,p)==False: #Computer plays
            p=-p #switch
            choice=Node(n,n,p) #create node where comp plays
            options=[]
            if len(choice.subtree)>0: #will ot touch terminal nodes
                for i in choice.subtree:
                    options=options+[reverse_deduction(i,-p)] #subtree starts with oppontent
            print("Current state: Terminal node score for computer\n"+str(options))
                        
            if min(options)==options[0]:
                if min(options)==options[1]:
                    print("computer loses either way. Computer plays randomly generated move")
                    computer_move=[1,2][random.randint(0,1)]
                else:
                    computer_move=1
            elif min(options)==options[1]:
                if min(options)==options[0]:
                    print("computer loses either way. Computer plays randomly generated move")
                    computer_move=[1,2][random.randint(0,1)]
                else:
                    computer_move=2
                                                              
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
