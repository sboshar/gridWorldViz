from mdp import MDP
from agent import Agent
from tabularq import TabularQ
import matplotlib.pyplot as plt
import numpy as np

def getSPrime(s, a):
  return (s[0] + a[0], s[1] + a[1])

#arrays of tuples signifying the positions
def setUpEnvironment(trapsPos, rewardPos, wallsPos, rows, cols):
  board = np.array([['' for c in range(cols)] for r in range(rows)])
  # board = {(r,c):'' for r in range(rows) for c in range(cols)}
  for t in trapsPos:
    board[t] = 't'
  for r in rewardPos:
    board[r] = 'r'
  for w in wallsPos:
    board[w] = 'w' 
  return board

def reward(s, a, rewards={'': -1, 'w': -1, 't': -100, 'r': 0}):
  sPrime = getSPrime(s, a)
  if not validMove(s, a):
    return -1
  return rewards[board[sPrime]]
    
def createTransition(probs, actions):
  transition = {}
  for i in range(len(probs)):
    transition[actions[i]] = probs[i]
  return transition

def transition(s, a):
  #check wall distribute the probability  
  total = 0
  for a in actions:
    if validMove(s, a):
      total += transitionDict[a]
  return transitionDict[a] / total

def validActions(s):
  validActions = []
  for a in actions:
    if validMove(s, a):
      validActions.append(a)
  return validActions


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



# def expectedValue2(s, q_table):
#     def value(s, q_table):
#       return max([q_table.getValue(s,a) for a in validActions(s)]) #q_table.validActions(s)



#     return sum([mdp.transition(s, a) * value(getSPrime(s, a), q_table) for a in validActions(s)])





def value_iteration(mdp, q_table, eps=0.01, max_iter=1000):
  #value of a particular 
  
  def expectedValue(s, q_table):
    def value(s, q_table):
      return max([q_table.getValue(s,a) for a in validActions(s)]) #q_table.validActions(s)

    # Only check if transition is not 0
    # print("top")
    # for a in validActions(s):
    #   print(s, a, validActions(s))
    #   print(mdp.transition(s, a), value(getSPrime(s, a), q_table))
    return sum([mdp.transition(s, a) * value(getSPrime(s, a), q_table) for a in validActions(s)])

  for i in range(max_iter):
    new_q_table = q_table.copy()
    delta = 0
    print(i)  
    
    for x in range(len(board)):
      for y in range(len(board[0])):
        s = (x, y)
        for a in q_table.actions:
          #only take action if possible
          #just dont take action if it leads you out of bounds
          # if validMove(s, a):
          if board[s] != 'w':
            print(s, a, mdp.reward(s,a), mdp.discount_factor * expectedValue(getSPrime(s,a), q_table))

            # print("SUM", expectedValue(s, q_table))
            new_q_table.setValue(s,a,mdp.reward(s,a) + mdp.discount_factor * expectedValue(getSPrime(s, a), q_table))
            delta = max(delta, abs(new_q_table.getValue(s,a) - q_table.getValue(s,a)))
    if delta < eps:
      return new_q_table
  
    q_table = new_q_table

  # mdp.processAction()

  return q_table
    
rows = 5
cols = 5
actions =  [(1, 0), (-1, 0), (0, -1), (0, 1)]
transitionDict = createTransition([.25, .25, .25, .25], actions)
# states = [(x, y) for x in range(cols) for y in range(cols)]
traps = [(4, 4)]
treasure = [(1, 1)]
walls = [(2, x) for x in range (1, 3)]
board = setUpEnvironment(traps, treasure, walls, rows, cols)
q_table = TabularQ(board, actions)
mdp = MDP(Agent(4, 4), board, actions, transition, reward, 0.9) 
final_q = value_iteration(mdp, q_table)
# print(final_q.q_table)


# Indices are messed up
bestActions = []
for x in range(cols):
  for y in range(rows):
    # print(final_q.q_table[(x, y)])
    bestActions.append(np.argmax(final_q.q_table[(x, y)]))

horiz = np.array([actions[i][0] for i in bestActions])
vert = np.array([actions[i][1] for i in bestActions])
# print(bestActions)
converted = [[max([final_q.getValue((x, y), a) for a in final_q.actions]) for y in range(cols)] for x in range(rows)]



mi = min([min(i) for i in converted])
for w in walls:
  converted[w[0]][w[1]] = mi
ma = max([max(i) for i in converted])
# print("CONVERTED --------------------------------")
# print(np.array(converted))
# print("-------------------------------")

converted = [[np.interp(converted[y][x], [mi, ma], [0, 1]) for y in range(rows)] for x in range(cols)]
# def convertToArr(dict):
#   for a in final_q.actions:
    

# [[q_table.getValue(i, z)) for i in range(rows)] for z in range(cols)]
X = np.arange(0, rows, 1)
Y = np.arange(0, cols, 1)
# print(horiz)
# print(vert)

X, Y = np.meshgrid(X, Y)
# print(X)
# print(Y)

fig, ax = plt.subplots()
q = ax.quiver(X, Y, horiz, vert)

plt.imshow(converted)
plt.colorbar()
plt.show()
