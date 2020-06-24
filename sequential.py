from tabularq import TabularQ
from board import Board
import matplotlib.pyplot as plt
import numpy as np
import random

class Sequential(object):

  def __init__(self, actions=[(-1, 0), (1, 0), (0, -1), (0, 1)], board=Board(rand=True), pTransition=0.7, plot=True):
    self.board = board
    self.actions = actions
    self.q_table = TabularQ(board.shape, actions)
    self.pTransition = pTransition
    self.eps = 0.01
    self.max_iter = 10000
    self.plot = plot
    self.plotEvery = False
    self.verbose = False
    self.recolor = 0.98
    self.arrows = True
    self.discount_factor = .95

  def transition(self, s, a):
    """ Returns a dictionary that maps the neighbors of state s to the probability of transitioning
        to the neighbors. The probability of transitioning to state s' by taking action a is 
        pTransition. 
        
    Args:
        s (tuple): state
        a (tuple): intended action
        pTransition (float, optional): Probability of transitioning to state s' by taking action a is 
                                      pTransition. 
                                      Defaults to 0.7.

    Returns:
        dict: maps neighboring states of s to the probability of transitioning to each neighbor
    """
    
    validNeighbors = [self.board.getSPrime(s, a) for a in self.actions if self.board.isValidState(self.board.getSPrime(s, a))]
    if len(validNeighbors) == 1: #Always transition to validNeighbor[0] if there is one valid neighbor
      return {validNeighbors[0] : 1}
    elif len(validNeighbors) == 0:
      return {(0,0) : 0}
    return {sPrime: self.pTransition if (self.board.getSPrime(s, a) == sPrime) else ((1 - self.pTransition) / (len(validNeighbors) - 1)) for sPrime in validNeighbors}

  def value(self, s):
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
    if s in self.board.traps: 
      return self.board.trapReward
    if s in self.board.treasures:
      return self.board.treasureReward
    values = [self.q_table.getValue(s, a) for a in self.actions if self.board.isValidState(self.board.getSPrime(s, a))]
    if values:
      return max(values)
    return 0 #Return 0 if a state has no valid neighbors

  def expectedValue(self, s, a, q_table):
    """ The expected future value of taking action a from state s. The expectedValue is a weighted
        sum of the values of the neighboring states of s. Taking action a from s is weighted more 
        heavily. 

    Args:
        s (tuple): state
        a (tuple): action
        q_table (TabularQ): q table object

    Returns:
        float: the expected value of taking action a from state s
    """
    return sum([self.value(sPrime) * p for (sPrime, p) in self.transition(s, a).items()]) 

  def value_iteration(self):
    """This is the value iteration algorithm. It updates the q_table by updating a values of a state
    to be the reward of leaving that state by taking action a plus the expected future rewards times a discount
    factor. The expected future rewards is a an expectation over the future possible states, and discount factor
    enforces that future rewards are valued less than immediate rewards. The algorithm is said to be complete 
    when the max difference between any q value of the state action pair in the current q table and the new q table
    is less than a parameter eps.

    Args:
        q_table (TabularQ): holds q values for state action pairs
        eps (float, optional): Defines the stopping criterion.
                              Defaults to 0.01.
        max_iter (int, optional): The maximum iterations of value_iteration.
                                  Defaults to 10000.
        plot (bool, optional): Plots the value array, and policy using arrows in matplotlib.
                              Defaults to False.
        plotEvery (bool, optional): Plots every iteration, click x to get next iteration.
                                    Defaults to False.
        verbose (bool, optional): print the delta between old and new q, and the array of values.
                                  Defaults to False.
        recolor (float, optional): a float in range [0,1) used for recoloring. 0 being no recoloring and 1 being
                                  maximum recoloring.
                                  Defaults to 0.98.
        arrows (bool, optional): Whether arrows are drawn on the board.
                                Defaults to True.

    Returns:
        TabularQ: the final q_table
    """
    for i in range(self.max_iter):
      new_q_table = self.q_table.copy()
      if self.plotEvery: self.board.plot(self.q_table, self.actions, verbose=self.verbose, recolor=self.recolor, arrows=self.arrows)
      delta = 0  
      for x in range(self.board.rows):
        for y in range(self.board.cols):
          s = (x, y)
          for a in self.actions:
            if self.board.isValidState(s):
              #update q value
              new_q_table.setValue(s,a, self.board.reward(s) + self.discount_factor * self.expectedValue(s,a, self.q_table))
              #define delta to me the max change in any q value
              delta = max(delta, abs(new_q_table.getValue(s,a) - self.q_table.getValue(s,a)))
      if self.verbose: 
        print("-------------------------------------------------------------------------")
        print(f"Iteration {i}: DELTA {delta}")
        print("-------------------------------------------------------------------------")
      #q_table is not changing 
      if delta < self.eps:
        if self.plot: self.board.plot(self.q_table, self.actions, verbose=self.verbose, recolor=self.recolor, arrows=self.arrows)
        return new_q_table
      self.q_table = new_q_table.copy()
    return self.q_table

if __name__ == "__main__":
  s = Sequential()
  s.value_iteration()
