import numpy as np
class TabularQ:

  def __init__(self, board, actions):
    self.board = board
    self.actions = actions

    self.__actionDict = {}
    for i in range(len(actions)):
      self.__actionDict[actions[i]] = i

    # print(type(board))
    #create a dictionary for the q q_table intialize q values to 0.0
    self.q_table = np.zeros(board.shape + (len(actions),))
    
  def getQ(self):
    return np.copy(self.q_table)
  
  def __mapAction(self, a): 
    return self.__actionDict[a]
  
  def getActions(self):
    return np.copy(self.actions)

  def setQ(self, new_q):
    #updates/sets the q table to be new_q expect dict
    self.q_table = new_q

  # def validActions(self, s):
  #   def validMove(s, a):
  #     sPrime = (s[0] + a[0], s[1] + a[1])
  #     #include walls
  #     if (s[0] + a[0] >= cols or s[0] + a[0] < 0):
  #       return False
  #     if (s[1] + a[1] >= rows or s[1] + a[1] < 0):
  #       return False
  #     if board[sPrime] == 'w':
  #       return False
  #     return True

    # validActions = []
    # for a in actions:
    #   if validMove(s, a):
    #     validActions.append(a)
    # return validActions

  def getValue(self, s, a):
    return self.q_table[s][self.__actionDict[a]]

  def setValue(self, s, a, v ):
    self.q_table[s][self.__actionDict[a]] = round(v, 5)

  def copy(self):
    new_q_table = TabularQ(self.board, self.actions)
    new_q_table.setQ(np.copy(self.q_table))\
    # print(id(new_q_table) == id(self.q_table))
    return new_q_table

 