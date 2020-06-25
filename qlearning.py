from board import Board
from gridworld import GridWorldEnv
from tabularq import TabularQ
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from temporalDifference import TemporalDifference

epsarray = []
#off policy Temporal difference learning
class QLearning(TemporalDifference):

  def __init__(self, num_epochs=1000, board=Board(rand=True), actions=[(-1, 0), (1, 0), (0, -1), (0, 1)], 
               minEps=0.1, minAlpha=0.1, gamma=0.9, max_steps=250):
    super().__init__(num_epochs, board, actions, minEps, minAlpha, gamma, max_steps)
    
  def update(self, s, a, sprime, reward, alpha, gamma):
    value = (1 - alpha) * self.q_table.getValue(s, a) + alpha * (reward + gamma * np.max(self.q_table.getQ()[sprime]))
    self.q_table.setValue(s, a, value)

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
        
        self.update(prevState, action, self.env.state, reward, alpha, self.gamma)
      epsarray.append(eps_fn(epoch))
      counts.append(count)
     
      #do we want to divide by count?
      rewards.append(epoch_reward)


    return rewards, counts
  
  def __str__(self):
    return "QLearning"
  
if __name__ == "__main__":
  agent = QLearning(num_epochs=10000)
  r, c = agent.run(agent.epsLinear, agent.epsLinear)
  print("Done")
  new = []
  print(agent.q_table.q_table.shape)
  print("-------------------")
  print(agent.q_table.q_table)
  print("-------------------")
  print(agent.q_table.getValueTable())
  agent.env.board.plot(agent.q_table, agent.actions, verbose=True, recolor=0.98)
  for i in range(len(r)):
    if i % 100:
      new.append(np.mean(r[i: i + 100]))
  print(len(c))
  #plt.plot(epsarray)
  #plt.plot(c)
  plt.plot(new)
  plt.show()
      
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
