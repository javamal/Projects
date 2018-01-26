import math, random, numpy as np

class Tree(object):
    def __init__(self,depth,p,board,score=0):
        self.depth = depth
        self.p = p
        self.board = board
        self.score = score
        self.node = []
        self.move = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8], [0,4,8],[2,4,6]]
        self.create_node()
        
    def create_node(self):
        if self.depth > 0 and len(np.where(self.board=="-")[0])!=0 and self.assign(self.board)==0: #no more depth, no more place to play, or game win/lost
            for i in np.where(self.board=="-")[0]:
                options = np.copy(self.board) #one move per time
                if self.p == 1:
                    options[i] = "x"
                    self.node.append(Tree(self.depth-1,-self.p,options, score = self.assign(options)) )
                else:
                    options[i] = "o"
                    self.node.append(Tree(self.depth-1,-self.p,options, score = self.assign(options)) )
    
    def assign(self, board):
        for i in self.move:
            if len(np.where(board[i]=="x")[0])==3:
                return(math.inf)
            elif len(np.where(board[i]=="o")[0])==3:
                return(-math.inf)
        else:
            return(0)

def reverse(tree,player):
    if len(tree.node)==0:
        return(tree.score)
    elif player==1:
        best=-math.inf
        for i in tree.node:
            new_best=reverse(i,-player)
            best=max(new_best,best)
        return(best)
    else:
        best=math.inf
        for i in tree.node:
            new_best=reverse(i,-player)
            best=min(new_best,best)
        return(best)
    
def game_over(board):
    move = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8], [0,4,8],[2,4,6]]
    if len(np.where(board=="-")[0])==0:
        print("draw")
        return(True)    
    for i in move:
            if len(np.where(board[i]=="x")[0])==3:
                print("human wins")
                return(True)
            elif len(np.where(board[i]=="o")[0])==3:
                print("computer wins")
                return(True)
    else:
        return(False)


def play():
    player = 1
    board = np.array(["-"]*9)
    depth = 8
    while True:
        
        while True:
            human = int(input("0-8: "))
            if human<0 or human>=9:
                print("out of range try again")
            elif board[human]!="-":
                print("taken try again")
            else:
                break
        board[human]="x"                            
        print(np.reshape(board,(3,3)))
        if game_over(board) == True:
            break
        player = -player
        computer_option = []
        computer_board = []
        computer = Tree(depth, player, board)
        if len(computer.node)>0:
            for i in computer.node:
                computer_option.append(reverse(i, -player))
                computer_board.append(i.board)
            print(computer_option)
        computer_coord = np.where(np.array(computer_option)==min(computer_option))[0]
        if len(computer_coord)>1:
            print("computer plays randomly generated move")
            computer = computer_coord[random.randint(0,len(computer_coord)-1)]
            board = computer_board[computer]
            print(np.reshape(board,(3,3)))
        else:
            computer = computer_coord[0]
            board = computer_board[computer]
            print(np.reshape(board,(3,3)))
        if game_over(board) == True:
            break
        else:
            player = -player
    play_again = input("play again? (y/n): ")
    if play_again == "y":
        play()
    else:
        return("game over")
            
play()
