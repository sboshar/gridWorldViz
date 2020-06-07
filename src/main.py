from tabularq import TabularQ
import matplotlib.pyplot as plt
import numpy as np
import random

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

def reward(s, a, rewards={'': 0, 'w': -1, 't': -5, 'r': 10}):
  #sPrime = getSPrime(s, a)    
  if not isValidMove(s, a):
    return -1
  #was sprime
  return rewards[board[s]]

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

def transition(s, a, pActual = 0.7):
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

def getAllValidStates():
  return [(x,y) for x in range(rows) for y in range(cols) if board[x][y] != 'w']
        

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
  if s in traps:
    return -5
  if s in treasure:
    return 3
  # rewards={'': 0, 'w': -.1, 't': -5, 'r': 3}
  return max([q_table.getValue(s, a) for a in validActions(s)])
  # return np.max(q_table.getQ()[s]) if isValidState(s) else 0

def expectedValue(s, a, q_table):
  # if isTerminal(s):
  #   print("Found terminal state at", s)
  #   return randomValue(q_table)
  #if isTerminal(getSPrime(s, a)):
    #return 0
  return sum([value(sPrime, q_table) * p for (sPrime, p) in transition(s, a).items()]) 

def plotQ(q_table):
  values = [[value((y,x), q_table) for x in range(rows)] for y in range (cols)]
  print(np.round(np.array(values), 2))
  #what we needd to do is aweighted average of q_table actions
  # converted = [[max([expectedValue((x, y), a, q_table) for a in validActions((x, y))]) for y in range(cols)] for x in range(rows)]
  # print(q_table.getQ()[(1,2)])
  # print(q_table.getQ()[(1,0)]) 
  # print(q_table.getQ()[(0,1)])
  # print(q_table.getQ()[(2,1)])

  bestActions = []
  for x in range(cols):
    for y in range(rows):
      s = (x, y)
      bestA = 0
      for a in range(len(actions)):
        if isValidState(getSPrime(s, actions[a])) and q_table.getValue(s, actions[a]) > q_table.getValue(s, actions[bestA]):
          bestA = a
      bestActions.append(bestA)
  # print("Best Actions")
  # print(np.array(bestActions).reshape(rows, cols))  
  
  bestActions1 = np.array(bestActions).reshape(rows, cols)


  pActual = .7
  converted = []
  for x in range(cols):
    temp = []
    for y in range(rows):
      weightedS = 0
      vActions = validActions(s)
      s = (x, y)
      for a in vActions:
        if a == actions[bestActions1[s]]:
          weightedS += pActual * q_table.getValue(s, a)
        else:
          weightedS += (1 - pActual) / (len(vActions) - 1)
      temp.append(weightedS)      
    converted.append(temp)
  
  # print(converted)
  #converted = [[max([q_table.getValue((x,y), a) for a in validActions((x, y))]) for y in range(cols)] for x in range(rows)]
  
  converted = values
  mi = min([min(i) for i in converted]) + 4
  for t in traps:
    print(converted[t[0]][t[1]])
    converted[t[0]][t[1]] += 4
  print(mi) 
  for w in walls:
    converted[w[0]][w[1]] = mi # 1, 0????
  ma = max([max(i) for i in converted])
  # print(np.array(converted).reshape(rows,cols))
  
  converted = [[np.interp(converted[x][y], [mi, ma], [0, 1]) for y in range(rows)] for x in range(cols)]
  directions = ["r", "l", "d", "u"]
  
  # Indices are messed up

 

  # for a in actions:
  #   print("Value for taking action", a, "from state (4, 4)", q_table.getValue((4, 4), a))
  horiz = np.array([actions[i][1] for i in bestActions])
  vert = np.array([-1 * actions[i][0] for i in bestActions])

  # print("CONVERTED --------------------------------")
  # print(np.array(converted))
  # print("-------------------------------")
  # printBoard(board)

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

def averageValues():
  total = 0
  totalValue = 0
  for x in range(rows):
    for y in range(cols):
      s = (x, y)
      for a in validActions(s):
        total += 1
        totalValue += value(s, a)
  return totalValue / total

def randomValue(q_table):
  return value(random.choice(getAllValidStates()), q_table) if getAllValidStates() else None

def isTerminal(s):
  return s in treasure

def value_iteration(q_table, eps=0.01, max_iter=10000):
  #value of a particular 
  # plotQ(q_table)
  for i in range(max_iter):
    new_q_table = q_table.copy()
    plotQ(new_q_table)
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
      #print("TRANSITION LEFT", transition((4, 4), (0, -1)))
      #print("TRANSITION RIGHT", transition((4, 4), (0, 1)))
      return new_q_table

    q_table = new_q_table.copy()
  # mdp.processAction()

  return q_table

def wallGenerator():
  
  newWalls = []
  
  def isValidState(s, newWalls):
    if (s[0] >= rows or s[0] < 0):
      return False
    if (s[1] >= cols or s[1] < 0):
      return False
    if s in newWalls:
      return False
    return True
  numWallGroup = random.randint(2,3)
  
  directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])

  for i in range(numWallGroup):
    curr = (random.randint(0, rows), random.randint(0, cols))
    #if no spots open this will be an issue
    failed = 0
    while curr in newWalls:
      curr = (random.randint(0, rows), random.randint(0, cols)) 
      if failed > rows * cols * rows:
        return #BROKEN
      failed +=1

    length = random.randint(5, 8) #Depend on map size later?
    for i in range(length):
      np.random.shuffle(directions)
      isCornered = True
      for i in range(len(directions)):
        temp = getSPrime(curr, tuple(directions[i]))
        if isValidState(temp, newWalls):
          isCornered = False
          break
      if isCornered:
        break
      
      if temp:
        newWalls += [temp]
        curr = temp
  
  return newWalls


#Global
actions =  [(1, 0), (-1, 0), (0, -1), (0, 1)]
# actions =  [(1, 1), (-2, 1), (0, -2), (0, 0)]

rows = 10
cols = 10
# traps = [(1, 1)]
traps = [(6, 5)]  
treasure = [(5, 5)]
# traps = traps + [(6, 7)]
walls = [(2,3)]
#walls = wallGenerator()

board = setUpEnvironment(traps, treasure, walls, rows, cols)
printBoard(board)
# walls = walls + [(3, x) for x in range(5, )]
# walls = walls + [(5, x) for x in range(5, 9)]
# traps = [(5, 5), (10, 9), (10, 8),(10, 7),(10, 6),(10, 5),(10, 4),(10, 3),(10, 2),(10, 1),(10, 0)]
# treasure = [] 
# walls = []


#print(board)

# testSetUpEnvironment()
# testIsValidMove()
# testValidActions()
# testReward()
# testValidStates()
# testTransition()

discount_factor = .95
# transitionDict = createTransition([.25, .25, .25, .25], actions)
# states = [(x, y) for x in range(cols) for y in range(cols)]
q_table = TabularQ(board, actions)
# mdp = MDP(Agent(4, 4), board, actions, transition, reward, 0.9) 
final_q = value_iteration(q_table)
# print(final_q.q_table)




# def convertToArr(dict):
#   for a in final_q.actions: