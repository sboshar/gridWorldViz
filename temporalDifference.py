
from board import Board
from gridworld import GridWorldEnv
from tabularq import TabularQ
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class TemporalDifference(object):
  def __init__(self, num_epochs=1000, board=Board(rand=True), actions=[(-1, 0), (1, 0), (0, -1), (0, 1)], 
               minEps=0.1, minAlpha=0.1, gamma=0.9, max_steps=250):
    self.num_epochs = num_epochs
    self.env = GridWorldEnv(board, actions)
    self.q_table = TabularQ(board.shape, actions)
    self.actions = actions
    self.minEps = minEps
    self.minAlpha = minAlpha
    self.gamma = gamma
    self.max_steps = max_steps
    self.board = board

  def update(self, s, a, sprime, reward, alpha, gamma):
    raise NotImplementedError
  #strats
  def greedy(self, state):
 
    # print(self.q_table.getQ()[state])
    # print(self.actions[np.argmax(self.q_table.getQ()[state])]) 

    return self.actions[np.argmax(self.q_table.getQ()[state])] # Returns the action that maximizes the q of a given state

  #functions for decreasing epsilon (log, exp, linear)
  #def epsLog(self, epoch):
    #return 
  
  def epsilonLogDecay(self, epoch):                                                       
    return max(self.minEps, min(1, 1.0 - math.log10((epoch  + 1) / 25)))
  
  def epsLinear(self, epoch):
    chance = 1 - abs(epoch)/1000
    return chance if chance > .1 else .1

  def epsGreedy(self, s, eps, epoch): #epsilon is the chance you act randomly
    # print(eps(epoch))
    if random.random() < eps(epoch):
  
      return self.env.randAction()

    return self.greedy(s)
    # return self.env.randAction() if random.random() < eps(epoch) else self.greedy(s) 
  
  def run(self, eps_fn, alpha_fn):
    rewards = []
    counts = []
    print(self.env.board)
    for epoch in range(self.num_epochs):
      #if epoch % 10 == 0: print(epoch)
      done = False
      #reset the agents state each epoch
      self.env.reset()
      alpha = alpha_fn(epoch)
      epoch_reward = 0
      count = 0
      while not done and count < self.max_steps:
        #add a limit for iterations
        prevState = self.env.state
        action = self.epsGreedy(self.env.state, eps_fn, epoch)
        reward, done = self.env.step(action)
        epoch_reward += reward
        count += 1
        
        self.updateQ(prevState, action, self.env.state, reward, alpha, self.gamma)
      epsarray.append(eps_fn(epoch))
      counts.append(count)
        #print(self.env.state)
      # print("End of Epoch -------------------------------")
      #if epoch % 50 == 0:
        #agent.env.board.plot(agent.q_table, agent.actions, verbose=True, recolor=0.6)

      rewards.append(epoch_reward/count)


    return rewards, counts
  
  def value(self, s, q_table):
    """ Returns the value of the state s in q_table. The value for a trap or treasure is the 
        reward for leaving that states (-5 and 3 respectively). The value for any other state s is 
        the q value that is maximized by taking some action a' from s. 

    Args:
        s (tuple): state
        q_table (TabularQ): q table object

    Returns:
        float: the value of state s
    """
    # The value of special spaces is their reward
    if s in self.env.board.traps: 
      return self.env.board.trapReward
    if s in self.env.board.treasure:
      return self.env.board.treasureReward
    values = [self.q_table.getValue(s, a) for a in self.actions if env.board.isValidState(getSPrime(s, a), self.env.board)]
    if values:
      return max(values)
    return 0 #Return 0 if a state has no valid neighbor
  
# if __name__ == "__main__":
#   agent = GridWorldAgent(num_epochs=10000)
#   r, c = agent.run(agent.epsLinear, agent.epsLinear)
#   print("Done")
#   new = []
#   print(agent.q_table.q_table.shape)
#   print("-------------------")
#   print(agent.q_table.q_table)
#   print("-------------------")
#   print(agent.q_table.getValueTable())
#   agent.env.board.plot(agent.q_table, agent.actions, verbose=True, recolor=0.98)
#   for i in range(len(r)):
#     if i % 100:
#       new.append(np.mean(r[i: i + 100]))
#   print(len(c))
#   #plt.plot(epsarray)
#   #plt.plot(c)
#   plt.plot(new)
#   plt.show()
