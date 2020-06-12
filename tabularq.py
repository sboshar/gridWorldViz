import numpy as np
class TabularQ:

  def __init__(self, board, actions):
    self.board = board
    self.actions = actions

    #dictionary mapping each action in actions to the index of that action
    self.__actionDict = {}
    for i in range(len(actions)):
      self.__actionDict[actions[i]] = i

    #creates a dictionary for the q_table, intializes q values to 0.0
    self.q_table = np.zeros(board.shape + (len(actions),))
    
  def getQ(self):
    return np.copy(self.q_table)
  
  def __mapAction(self, a): 
    return self.__actionDict[a]
  
  def getActions(self):
    return np.copy(self.actions)

  def setQ(self, new_q):
    self.q_table = new_q

  def getValue(self, s, a):
    return self.q_table[s][self.__actionDict[a]]

  def setValue(self, s, a, v ):
    # rounds the value to avoid floating point errors
    self.q_table[s][self.__actionDict[a]] = round(v, 5)

  def copy(self):
    new_q_table = TabularQ(self.board, self.actions)
    new_q_table.setQ(np.copy(self.q_table))
    return new_q_table

 