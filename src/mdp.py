import numpy as np
import random

class MDP:
  def __init__(self, agent, board, actions, transition, reward, discount_factor):
    self.agent = agent
    self.board = board
    self.actions = actions
    self.transition = transition
    self.reward = reward
    self.discount_factor = discount_factor
  
  def getSPrime(self, s, a):
    return (s[0] + a[0], s[1] + a[1])

  #def greedy():
  
  def processAction(self, s, a):
    rows = board[(-1, -1)][0]
    cols = board[(-1, -1)][1]
    sPrime = self.getSPrime(s, a)
    if (board[s] == 'r'):
      newX = random.randint(0, cols - 1)
      newY = random.randint(0, rows - 1)
      #go to random
      self.agent.setX(newX)
      self.agent.setY(newY)      
      return
    if (self.board[sPrime] == 'w'):
      return 
    if (s[0] + a[0] >= cols or s[0] + a[0] < 0):
      return
    if (s[1] + a[1] >= rows or s[1] + a[1] < 0):
      return
    self.agent.move(s, a)
      

        


