from board import Board
from gridworld import GridWorldEnv
from tabularq import TabularQ
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class GridWorldAgent:

  def __init__(self, num_epochs=10000, board=Board(rand=True), actions=[(-1, 0), (1, 0), (0, -1), (0, 1)], 
               minEps=0.1, minAlpha=0.1, gamma=0.9):
    self.num_epochs = num_epochs
    self.env = GridWorldEnv(board, actions)
    self.q_table = TabularQ(board.shape, actions)
    self.actions = actions
    self.minEps = minEps
    self.minAlpha = minAlpha
    self.gamma = gamma
    
  #strats
  def greedy(self, state):
    return self.actions[np.argmax(self.q_table.getQ()[state])] # Returns the action that maximizes the q of a given state

  #functions for decreasing epsilon (log, exp, linear)
  def epsLog(self, epoch):
    return 
  def epsilonLogDecay(self, ep):                                                       
    return max(self.minEps, min(1, 1.0 - math.log10((ep  + 1) / 25)))
  
  def epsLinear(self, epoch):
    chance = 1 - abs(epoch)/1000
    return chance if chance > .1 else .1

  def epsGreedy(self, s, eps, epoch): #epsilon is the chance you act randomly
    return self.env.randAction() if random.random() < eps(epoch) else self.greedy(s) 

  def updateQ(self, s, a, sprime, reward, alpha, gamma):
    value = (1 - alpha) * self.q_table.getValue(s, a) + alpha * (reward + gamma * np.max(self.q_table.getQ()[sprime]))
    self.q_table.setValue(s, a, value)
  
  def run(self, eps_fn, alpha_fn):
    rewards = []
    print(self.env.board)
    for epoch in range(self.num_epochs):
      #if epoch % 10 == 0: print(epoch)
      done = False
      #reset the agents state each epoch
      self.env.reset()
      alpha = alpha_fn(epoch)
      epoch_reward = 0
      while not done:
        #add a limit for iterations
        prevState = self.env.state
        action = self.epsGreedy(self.env.state, eps_fn, epoch)
        reward, done = self.env.step(action)
        epoch_reward += reward
        self.updateQ(prevState, action, self.env.state, reward, alpha, self.gamma)
      
        #print(self.env.state)
      #print("End of Epoch -------------------------------")
      rewards.append(epoch_reward)
    return rewards
  
  def value(s, q_table):
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
    if s in traps: 
      return trapReward
    if s in treasure:
      return treasureReward
    values = [q_table.getValue(s, a) for a in actions if isValidState(getSPrime(s, a), board)]
    if values:
      return max(values)
    return 0 #Return 0 if a state has no valid neighbor
  
  def plotQ(q_table, verbose=False, recolor=0.98, arrows=True):
    """ Plots the values of each state on the board using matPlotLib. The board is colored with 
        a gradient to distinguish between high and low valued states, and arrows are plotted on 
        each state that point to the highest valued neighbor. Following the arrows, 
        therefore, represents, following the policy. 

    Args:
        q_table (TabularQ): [description]
        verbose (bool, optional): a boolean flag. If True, PlotQ will also print out the q_table 
                                  and the delta between the old q and the new q table. 
                                  Defaults to False.
        recolor (float, optional): a float in range [0,1) used for recoloring. 0 being no recoloring and 1 being
                                  maximum recoloring.
                                  Defaults to 0.98.
        arrows (bool, optional): Whether arrows are drawn on the board.
                                Defaults to True.
    """
    #a 2d array holding the values for each state
    values = [[value((y,x), q_table) for x in range(cols)] for y in range (rows)]
    
    if verbose: 
      print("-------------------------------------------------------------------------")
      print("Value Array:")
      print(np.round(np.array(values), 3))
      print("-------------------------------------------------------------------------")

    #min and max value of values used for caling the color purposes
    maxValue = max([max(i) for i in values])

    newMin = values[traps[0][0]][traps[0][1]]
    #if you are recoloring, set new min to be slightly higher than the second lowest value
    if recolor and traps:
      minValue = 0
      for y in range(rows):
        for x in range(cols):
          if (not (y, x) in traps) and minValue > values[y][x]:
            minValue = values[y][x]

      newMin = minValue + (values[traps[0][0]][traps[0][1]] - minValue) * (1 - recolor)
      for t in traps:
        values[t[0]][t[1]] = newMin
        
    #colors the walls as the darkest color on the plot
    for w in walls:
      values[w[0]][w[1]] = newMin
    
    #map to 0-1 range
    values = [[np.interp(values[x][y], [newMin, maxValue], [0, 1]) for y in range(cols)] for x in range(rows)]
    
    #creates an array of shape [1, row*col] where each element represents the best action to take in that state
    if arrows:
      bestActions = []
      for x in range(rows):
        for y in range(cols):
          s = (x, y)
          bestA = 0
          if not isValidState(s, board):
            bestActions.append(-1)
            continue
          for a in range(len(actions)):
            if isValidState(getSPrime(s, actions[a]), board): #find a valid best action, bestA
              bestA = a
              break
          for a in range(len(actions)): #compare each valid action to bestA
            if isValidState(getSPrime(s, actions[a]), board) and \
              q_table.getValue(s, actions[a]) >= q_table.getValue(s, actions[bestA]):
              bestA = a
          bestActions.append(bestA)
      
      horiz = np.array([actions[i][1] if i != -1 else 0 for i in bestActions])
      vert = np.array([-1 * actions[i][0] if i != -1 else 0 for i in bestActions]) 
      
      Y = np.arange(0, rows, 1)
      X = np.arange(0, cols, 1)

      X, Y = np.meshgrid(X, Y)
      _ , ax = plt.subplots()
      ax.quiver(X, Y, horiz, vert)
    plt.imshow(values)
    plt.colorbar()
    plt.show()

  if __name__ == "__main__":
    b = Board(walls=[(1,2), (1,3), (1,4)], treasures=[(5,5)], traps=[(4,4)])
    agent = GridWorldAgent(board=b)
    r = agent.run(agent.epsilonLogDecay, lambda epoch: .1)
    print("Done")
    print(agent.env.board)
    new = []
    for i in range(len(r)):
      if i % 100:
        new.append(np.mean(r[i: i + 100]))
    plt.plot(new)
    # plt.plot(r)
    plt.show()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

