class GridWorldEnv:
  def __init__(self):
    #some sort of board
    self.board = None
    #(x,y) coordinate on the board
    self.state = None

  
  def reset(self):
    self.state = self.board.getRandomState()
    return self.state

  def render(self): #Front end
    #this function will render a the agent moving in the environment
    raise NotImplementedError
  
  def step(self, action): 
    #return nextstate, reward, done
    nextState = self.board.getNextState(self.state, action)
    return nextState, self.board.reward(self.state), self.board.isTerminal(nextState)

if __name__ == '__main__':
  g = GridWorldEnv()
