import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
from torch import optim
from board import Board
#do i need to reduce the observation space dimensions?
#could have it take in multiple n-hot vector encodings one for treasures, one for walls, one for traps
#maybe being with it take in the board?
#gets the agents position as a one hot vector?

#begin with an implementation where the state space does not change
def boardToState(agentPos, board):
  new = []
  agent = np.array([0,0,0,1])
  wall = np.array([0,0,1,0])
  trap = np.array([0,1,0,0])
  reward = np.array([1,0,0,0])  
  empty = np.array([0,0,0,0])

  for x in range(len(board)):
    for y in range(len(board[0])):
      s = (x,y)
      if s == agentPos:
        new.append(agent)
      elif board[s] == 'w':
        new.append(wall)
      elif board[s] == 't':
        new.append(trap)
      elif board[s] == 'r':
        new.append(reward)
      elif board[s] == ' ':
        new.append(empty)
  return np.array(new)



class policy_estimator(nn.Module):
    def __init__(self):
        super(policy_estimator, self).__init__()
        self.n_inputs = 16
        self.n_outputs = 4
        
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 16), 
            nn.ReLU(), 
            nn.Linear(16, self.n_outputs),
            nn.Softmax(dim=-1))
    
    def forward(self, state):
        action_probs = self.network(torch.FloatTensor(state))
        return action_probs

if __name__ == "__main__":
  env = Board(rows = 2, cols = 2, walls = [(1,1)], traps = [(0,1)], treasures=[(1,0)])
  s = boardToState((0,0), env.board)
  print(s)
  print(s.reshape((2,2,4)))
  s = s.reshape((1,16))
  p = policy_estimator()
  print(p(s))
