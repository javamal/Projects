#working version
import math, numpy as np, random, time

class Node(object):
    def __init__(self,depth,board,player,score=0):
        self.depth=depth
        self.board=board
        self.player=player
        self.score=score
        self.subtree=[]
        self.create_subtree()
        
    def create_subtree(self):
        if self.depth>0 and self.win(self.board,self.player)==0 and len(np.where(self.board==0)[0])>0:
            for i in np.where(self.board==0)[0]:
                new=np.copy(self.board)
                if self.player==1:
                    new[i]=1
                    human=self.win(new,self.player)
                    self.subtree.append(Node(self.depth-1,new,-self.player,human) )
                else:
                    new[i]=-1
                    computer=self.win(new,self.player)
                    self.subtree.append(Node(self.depth-1,new,-self.player,computer) )
           
    def win(self,board,player):
        if player==1:
            my_move=1;counter_move=-1
        else:
            my_move=-1;counter_move=1
        moves=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for i in moves:
            if sum(board[i]==my_move)==3 or sum(board[i]==counter_move)==3:
                if player==1:
                    return(math.inf)
                else:
                    return(-math.inf)                
            '''
            else:
                for j in range(len(i)):
                    left=i[j]
                    if board[left]==0 and sum(board[i]==counter_move)==2:
                        return(-player*10)
            '''
        return(0) #when nobody wins

def reverse(node,player,depth,alpha=-math.inf,beta=math.inf):
    if len(node.subtree)==0 or depth==0:
        return(node.score)
    elif player==1:
        best=-math.inf
        for i in node.subtree:
            new_best=reverse(i,-player,depth-1,alpha,beta)            
            best=max(new_best,best)
            alpha=max(best,alpha)
            if beta <= alpha:
                break
        return(best)
    else:
        best=math.inf
        for i in node.subtree:
            new_best=reverse(i,-player,depth-1,alpha,beta)
            best=min(new_best,best)
            beta=min(best,beta)
            if beta <= alpha:
                break
        return(best)
        
def minmax(node,player):
    if len(node.subtree)==0:
        return(node.score)
    elif player==1:
        best=-math.inf
        for i in node.subtree:
            new_best=reverse(i,-player)            
            best=max(new_best,best)
        return(best)
    else:
        best=math.inf
        for i in node.subtree:
            new_best=reverse(i,-player)
            best=min(new_best,best)
        return(best)
        
#test
def test_time():
    board=np.array([0,0,0,0,0,0,0,0,0])
    a=Node(9,board,1,0)
    print("testing time for minmax")
    minmax.start=time.time()
    minmax(a,1)
    minmax.end=time.time()
    print("minmax algorithm takes: "+str(minmax.end-minmax.start)+" seconds.")
    
    print("testing time for alpha-beta pruning")
    reverse.start=time.time()
    reverse(a,1)
    reverse.end=time.time()
    print("alpha beta pruning algorithm takes: "+str(reverse.end-reverse.start)+" seconds.")
#test_time()
        
                
def check(player,board):
    if player==1 and win(board)==1:
        print("human wins")
        return(True)
    elif player==-1 and win(board)==2:
        print("computer win")
        return(True)
    else:
        return(False)

def win(board):
    moves=[[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for i in moves:
        if sum(board[i]==1)==3:
            return(1)
        elif sum(board[i]==-1)==3:
            return(2)
    return(0) #when nobody won
    
def play():
    board=np.array([0]*9)
    player=1
    depth=8
    while True:
        human_move=input("0 to 8: ")
        if board[int(float(human_move))]==1 or board[int(float(human_move))]==-1:
            return("cannot play")
        else:
            board[int(float(human_move))]=1
        print(np.reshape(board,(3,3)))
        if check(player,board)==True:
            again=input("play again? (y/n)" )
            if again=="y":
                play()
            else:
                return("end game")
        if len(np.where(board==0)[0])==0:
            return("draw")
        computer_option=np.array([])
        player=-player
        computer_board=[]
        for i in Node(depth-1,board,player).subtree:
            computer_option=np.append(computer_option,reverse(i,-player,depth-1))
            computer_board=computer_board+[i.board]
        print("Expected score for computer")
        print(computer_option) 
        if len(computer_option)==0:
            return("no place to move")
        domain=np.where(computer_option==min(computer_option))[0]
        if len(domain)>1:
            print("multiple moves with same value")
            board=computer_board[domain[random.randint(0,len(domain)-1)]]
        else:
            board=computer_board[domain[0]]
        print(np.reshape(board,(3,3)))
        if check(player,board)==True:
            again=input("play again? (y/n)" )
            if again=="y":
                play()
            else:
                return("end game")
        if len(np.where(board==0)[0])==0:
            return("draw")
        player=-player
play()    
