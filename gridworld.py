from board import Board
import random

class GridWorldEnv:
  def __init__(self, board, actions, eps=1.0):
    #some sort of board
    self.actions = actions
    self.board = board
    self.eps = eps
    #(x,y) coordinate on the board
    self.state = None

  #worry about seeds later
  def reset(self):
    self.state = self.board.getRandomState()
    return self.state

  def render(self): #Front end
    #have option to plot on top of a board that shows the values
    #this function will render a the agent moving in the environment
    raise NotImplementedError
  
  def randAction(self):
    return random.choice(self.actions)

  def step(self, action): 
    #return nextstate, reward, done
    reward = -1
    terminal = self.board.isTerminal(self.state)
    if random.random() < 1 - self.eps:
      action = self.randAction()
    
    if self.board.isValidState(self.board.getSPrime(self.state, action)):
      reward = self.board.reward(self.state)
      self.state = self.board.getSPrime(self.state, action)
  
    return reward, terminal
      
    
# if __name__ == '__main__':
#   b = Board(rand=True)
#   g = GridWorldEnv() 
