#q(s,a) = q(s,a) = + alpha(r + gammaQ(s,a) - Q(s,a))
from temporalDifference import TemporalDifference
from board import Board
from gridworld import GridWorldEnv
from tabularq import TabularQ
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class Sarsa(TemporalDifference):
  def __init__(self, num_epochs=1000, board=Board(rand=True), actions=[(-1, 0), (1, 0), (0, -1), (0, 1)], 
               minEps=0.1, minAlpha=0.1, gamma=0.9, max_steps=250):
    super().__init__(num_epochs, board, actions, minEps, minAlpha, gamma, max_steps)

  
  def update(self, s, a, sPrime, aPrime, reward, alpha, gamma):
    
    predict = self.q_table.getValue(s,a)
    target = reward + gamma * self.q_table.getValue(sPrime, aPrime)
    value = self.q_table.getValue(s,a) + alpha * (target - predict) 
    self.q_table.setValue(s, a, value)
    

  def run(self, eps_fn, alpha_fn):
    rewards = []
    counts = []
    for epoch in range(self.num_epochs):
      #if epoch % 10 == 0: print(epoch)
      done = False
      #reset the agents state each epoch
      s = self.env.reset()
      a = self.epsGreedy(self.env.state, eps_fn, epoch)
      alpha = alpha_fn(epoch)
      epoch_reward = 0
      count = 0
      while not done and count < self.max_steps:

        reward, done = self.env.step(a)

        aPrime = self.epsGreedy(self.env.state, eps_fn, epoch)
        
        self.update(s, a, self.env.state, aPrime, reward, alpha, self.gamma)

        a = aPrime
        s = self.env.state

        epoch_reward += reward
        count += 1
      
      counts.append(count)
      #do we want to divide by count?
      rewards.append(epoch_reward)
    return rewards, counts

  def __str__(self):
    return "Sarsa"

if __name__ == "__main__":
  agent = Sarsa(num_epochs=10000)
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
  plt.plot(r)
  plt.show()
  