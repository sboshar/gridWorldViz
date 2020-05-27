from mdp import MDP
from agent import Agent
from tabularq import TabularQ
import matplotlib.pyplot as plt
import numpy as np

def getSPrime(s, a):
  return (s[0] + a[0], s[1] + a[1])

#arrays of tuples signifying the positions
def setUpEnvironment(trapsPos, rewardPos, wallsPos, rows, cols):
  board = {(r,c):'' for r in range(rows) for c in range(cols)}
  for t in trapsPos:
    board[t] = 't'
  for r in rewardPos:
    board[r] = 'r'
  for w in wallsPos:
    board[w] = 'w' 
  
  board[-1, -1] = (rows, cols) # (-1, -1) maps to board dimensions
  return board

def reward(s, a, rewards={'': -1, 'w': -1, 't': -100, 'r': 0}):
  return rewards[board[getSPrime(s, a)]]
    
def createTransition(probs, actions):
  transition = {}
  for i in range(len(probs)):
    transition[actions[i]] = probs[i]
  return transition

def transition(s, a):
  #check wall 
  total = 0
  for a in actions:
    if validMove(s, a):
      total += transitionDict[a]
  return transitionDict[a] / total

def validMove(s, a):
  sPrime = getSPrime(s, a)
  #include walls
  if (s[0] + a[0] >= cols or s[0] + a[0] < 0):
    return False
  if (s[1] + a[1] >= rows or s[1] + a[1] < 0):
    return False
  if board[sPrime] == 'w':
    return False
  return True

def value_iteration(mdp, q_table, eps=0.01, max_iter=100):
  #value of a particular 
  
  def expectedValue(s):
    def value(s):
      return max([q_table.getValue(s,a) for a in q_table.actions])
    # Only check if transition is not 0
    return sum([mdp.transition(s, a) * value(getSPrime(s, a)) for a in q_table.actions if validMove(s, a)])

  for i in range(max_iter):
    new_q_table = q_table.copy()
    delta = 0
    print(i)
    for s in q_table.states:
      for a in q_table.actions:
        #only take action if possible
        #just dont take action if it leads you out of bounds
        if validMove(s, a):
          new_q_table.setValue(s,a,mdp.reward(s,a) + mdp.discount_factor * expectedValue(s))
          delta = max(delta, abs(new_q_table.getValue(s,a) - q_table.getValue(s,a)))
    if delta < eps:
      return new_q_table
    q_table = new_q_table

  # mdp.processAction()

  return q_table
    
rows = 100
cols = 100
actions =  [(1, 0), (-1, 0), (0, -1), (0, 1)]
transitionDict = createTransition([.25, .25, .25, .25], actions)
states = [(x, y) for x in range(cols) for y in range(cols)]
q_table = TabularQ(states, actions)
board = setUpEnvironment([(49, 49), (91, 90), (40, 40)], [(90, 90), (10, 60), (99, 40)], [(5, 5), (7, 7)], rows, cols)
mdp = MDP(Agent(4, 4), board, actions, transition, reward, 0.999) 
final_q = value_iteration(mdp, q_table)
print(final_q.q_table)

converted = [[max([final_q.getValue((x, y), a) for a in final_q.actions]) for x in range(cols)] for y in range(rows)]

ma = max([max(i) for i in converted])
mi = min([min(i) for i in converted])
converted = [[np.interp(converted[x][y], [mi, ma], [0, 1]) for x in range(rows)] for y in range(cols)]
# def convertToArr(dict):
#   for a in final_q.actions:
    

# [[q_table.getValue(i, z)) for i in range(rows)] for z in range(cols)]
plt.imshow(converted)
plt.colorbar()
plt.show()
