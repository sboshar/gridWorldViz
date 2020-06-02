from mdp import MDP
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

def testSetUpEnvironment():
  print("Testing setUpEnvironment--------------")
  board = setUpEnvironment([(1, 1)], [(0, 0), (2, 2)], [], 5, 5)
  printBoard(board)
  print("---------------------------")

def reward(s, a, rewards={'': -.1, 'w': -.1, 't': -10, 'r': 3}):
  sPrime = getSPrime(s, a)    
  if not isValidMove(s, a):
    return -.1
  return rewards[board[sPrime]]

def testReward():
  print("Testing reward ------------------")
  print("Expecting -100:", reward((1, 0), (0, 1)))
  print("Expecting -1:", reward((0, 1), (3, 0)))
  print("Expecing 0", reward((2, 3), (1, 0)))
  print("-------------------")

# def createTransition(probs, actions):
#   transition = {}
#   for i in range(len(probs)):
#     transition[actions[i]] = probs[i]
#   return transition

def transition(s, a, pActual = 0.5):
  # sPrime = getSPrime(s, a)
  # possibleStates = validStates(s)
  # if len(possibleStates) == 0:
  #   return {}
  # if len(possibleStates) == 1:
  #   return {possibleStates[0]: 1}
  expectedState = getSPrime(s, a)
  neighbors = validStates(s)
  # print(neighbors, s, a, end=" ")

  return {sPrime: pActual if (getSPrime(s, a) == sPrime) else ((1 - pActual) / (len(neighbors) - 1)) for sPrime in neighbors}
  # return {getSPrime(s, aPrime): pActual if (expectedState == getSPrime(s, aPrime)) else (1 - pActual) / (len(actions) - 1) for aPrime in actions}

  # return {validS: pActual if (getSPrime(s,a) == validS) else ((1 - pActual) / (len(possibleStates) - 1)) for validS in validStates(s)}

def testTransition():
  print("Testing transition")
  print("Expecting (1, 0) --> .9, (0, 1) --> .1")
  d = transition((0, 0), (1, 0))
  for a, b in d.items():
    print(a, b)

  print("Expecting .9 and even distribution on the other three")
  d = transition((3, 1), (1, 0))
  for a, b in d.items():
    print(a, b)

  print("Expecting: ")
  d = transition((2, 4), (1, 0))
  for a, b in d.items():
    print(a, b)

  print("Expecting: ")
  d = transition((2, 4), (0, 1))
  for a, b in d.items():
    print(a, b)


    
  # transition((0, 0), (1, 0)))

# def testTransition(s, a, pActual):

  
# def transition(s, a):
#   #check wall distribute the probability  
#   total = 0
#   for a in actions:
#     if isValidMove(s, a):
#       total += transitionDict[a]
#   return transitionDict[a] / total

def validActions(s):
  validActions = []
  for a in actions:
    if isValidMove(s, a):
      validActions.append(a)
  return validActions

def testValidActions():
  print("Testing validActions -------------------")
  print("Expecting (1, 0), (0, 1):", validActions((0, 0)))
  print("Expecting (-1, 0), (0, 1):", validActions((4, 0)))
  print("--------------------------------")

def isValidMove(s, a):
  return isValidState(getSPrime(s, a))
 
def isValidState(s):
  if (s[0] >= rows or s[0] < 0):
    return False
  if (s[1] >= cols or s[1] < 0):
    return False
  if board[s] == 'w':
    return False
  return True


def testIsValidMove():
  print("Testing isValidMove -----------------")
  print("Expecting False False False True")
  print(isValidMove((0, 0), (-1, 0)))
  print(isValidMove((4, 4), (1, 0)))
  print(isValidMove((3, 0), (2, 0)))
  print(isValidMove((0, 3), (1, 1)))
  print("-------------------------")

def validStates(s):
  return [getSPrime(s, a) for a in actions if isValidMove(s, a)]

def testValidStates():
  print("Testing valid state----------------")
  print("Expecting: (1, 4), (3, 4), (1,0), (0,1)")
  print(validStates((2,4)))
  print(validStates((0,0)))
  print("-----------------")

# def expectedValue2(s, q_table):
#     def value(s, q_table):
#       return max([q_table.getValue(s,a) for a in validActions(s)]) #q_table.validActions(s)

#     return sum([mdp.transition(s, a) * value(getSPrime(s, a), q_table) for a in validActions(s)])


def printBoard(board):
  print("BOARD -------------------")
  print(board)
  print("--------------------------")

def value(s, q_table):
  return max([q_table.getValue(s, a) for a in validActions(s)])
  # return np.max(q_table.getQ()[s]) if isValidState(s) else 0

def expectedValue(s, a, q_table, pActual = 0.5):
  # print("SUMMMMM of probabilities: " + str(sum(transition(s, a, pActual).values())))
  # print("Transition of " + str(s) + ", " + str(a) +  ": " + str(transition(s, a, pActual)))
  # print(len(validStates(s)), sum([value(sPrime, q_table) * p for (sPrime, p) in transition(s, a, pActual).items()]))
  return sum([value(sPrime, q_table) * p for (sPrime, p) in transition(s, a, pActual).items()]) 

def plotQ(q_table):
  converted = [[max([q_table.getValue((x, y), a) for a in validActions((x, y))]) for y in range(cols)] for x in range(rows)]
  mi = min([min(i) for i in converted])
  for w in walls:
    converted[w[0]][w[1]] = mi # 1, 0????
  ma = max([max(i) for i in converted])
  # print(np.array(converted).reshape(rows,cols))
  converted = [[np.interp(converted[x][y], [mi, ma], [0, 1]) for y in range(rows)] for x in range(cols)]
  
  directions = ["r", "l", "d", "u"]

  # Indices are messed up
  bestActions = []
  for x in range(cols):
    for y in range(rows):
      s = (x, y)
      bestA = 0
      for a in range(len(actions)):
        if isValidState(getSPrime(s, actions[a])) and q_table.getValue(s, actions[a]) > q_table.getValue(s, actions[bestA]):
          bestA = a
      bestActions.append(bestA)
  print("Best Actions")
  print(np.array(bestActions).reshape(rows, cols))  

  for a in actions:
    print("Value for taking action", a, "from state (8, 8)", q_table.getValue((8, 8), a))
  horiz = np.array([actions[i][1] for i in bestActions])
  vert = np.array([-1 * actions[i][0] for i in bestActions])

  print("CONVERTED --------------------------------")
  print(np.array(converted))
  print("-------------------------------")
  printBoard(board)

  # def printBudgetArrows():
  #   print("DIRECTIONS-----------------")
  #   for r in np.array(bestActions).reshape(rows, cols):
  #     for tile in r:
  #       print(directions[tile], end ="")
  #     print("")
  #   print("-----------------------")

  # printBudgetArrows()
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

  # , cmap=plt.get_cmap("YlGnBu")
  plt.imshow(converted)
  plt.colorbar()
  plt.show()

def value_iteration(q_table, eps=0.01, max_iter=10000):
  #value of a particular 
  # plotQ(q_table)
  for i in range(max_iter):
    # print(i)
    new_q_table = q_table.copy()
    delta = 0  
    for x in range(len(board)):
      for y in range(len(board[0])):
        s = (x, y)
        for a in actions:
          #only take action if possible
          #just dont take action if it leads you out of bounds
          if isValidMove(s, a):
          # if board[s] != 'w':
            # if isValidMove(s, a):
            new_q_table.setValue(s,a,reward(s,a) + discount_factor * expectedValue(s,a, q_table))
            delta = max(delta, abs(new_q_table.getValue(s,a) - q_table.getValue(s,a)))
            # else:
            #   new_q_table.setValue(s,a, q_table.getValue(s, a) - 1)
    print("DELTA", delta)
    if delta < eps:
      plotQ(new_q_table)
      return new_q_table

    q_table = new_q_table.copy()
  # mdp.processAction()

  return q_table


#Global
actions =  [(1, 0), (-1, 0), (0, -1), (0, 1)]
# actions =  [(1, 1), (-2, 1), (0, -2), (0, 0)]

rows = 10
cols = 10
# traps = [(1, 1)]
traps = [(1, 2)]
treasure = [(4, 7)]
traps = traps + [(6, 7)]
walls = [(3, 4), (5, 4)]
# walls = walls + [(3, x) for x in range(5, )]
# walls = walls + [(5, x) for x in range(5, 9)]
# traps = [(5, 5), (10, 9), (10, 8),(10, 7),(10, 6),(10, 5),(10, 4),(10, 3),(10, 2),(10, 1),(10, 0)]
# treasure = [] 
# walls = []

board = setUpEnvironment(traps, treasure, walls, rows, cols)
print(board)

# testSetUpEnvironment()
# testIsValidMove()
# testValidActions()
# testReward()
# testValidStates()
# testTransition()

discount_factor = .99
# transitionDict = createTransition([.25, .25, .25, .25], actions)
# states = [(x, y) for x in range(cols) for y in range(cols)]
q_table = TabularQ(board, actions)
# mdp = MDP(Agent(4, 4), board, actions, transition, reward, 0.9) 
final_q = value_iteration(q_table)
# print(final_q.q_table)




# def convertToArr(dict):
#   for a in final_q.actions:
    

