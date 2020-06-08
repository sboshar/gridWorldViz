from tabularq import TabularQ
import matplotlib.pyplot as plt
import numpy as np
import random

def getSPrime(s, a):
  """ Returns the state you would travel to taking action a from state s

  Args:
      s (tuple): state
      a (tuple): action

  Returns:
      tuple: new state, s'
  """
  return (s[0] + a[0], s[1] + a[1])
  
#arrays of tuples signifying the positions
def setUpEnvironment(trapsPos, rewardPos, wallsPos, rows, cols):
  """Takes arrays of tuples containing the loactions of the board 
  elements includuing traps, reward etc, and creates a board of dim row by cols
  with those features.

  Parameters
  ----------
  trapsPos (array): array of tuples for trap pos
  rewardPos (array): array of tuples for reward pos
  wallsPos (array): array of tuples for wall pos
  rows (int): num of rows in board env
  cols (int): num of col in board en

  Returns
  -------
  numpy array
      returns the initalized board as a numpy array
  """
  board = np.array([['' for c in range(cols)] for r in range(rows)])
  for t in trapsPos:
    board[t] = 't'
  for r in rewardPos:
    board[r] = 'r'
  for w in wallsPos:
    board[w] = 'w' 
  return board

def reward(s, rewards={'': 0, 'w': -1, 't': -5, 'r': 10}):
  """ returns the reward for leaving the current state, s
  Args:
      s (tuple): state
      rewards (dict, optional): Dictionary that maps board positions (strings) totheir reward (ints) 
                                Defaults to {'': 0, 'w': -1, 't': -5, 'r': 10}.

  Returns:
      tuple: [description]
  """
  
  # if not isValidMove(s, a): # Neccesarry?
  #   return -1
  return rewards[board[s]]

def transition(s, a, pTransition = 0.7):
  """ returns a dictionary that maps the neighbors of state s to the probability of transitioning
      to the neighbors. The probability of transitioning to state s' by taking action a is 
      pTransition. 
      
  Args:
      s (tuple): state
      a (tuple): intended action
      pTransition (float, optional): Probability of transitioning to state s' by taking action a is 
                                     pTransition. Defaults to 0.7.

  Returns:
      dict: maps neighboring states of s to the probability of transitioning to each neighbor
  """
  validNeighbors = validStates(s)
  return {sPrime: pTransition if (getSPrime(s, a) == sPrime) else ((1 - pTransition) / (len(validNeighbors) - 1)) for sPrime in validNeighbors}
  
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

def validStates(s):
  return [getSPrime(s, a) for a in actions if isValidMove(s, a)]

def value(s, q_table):  
  if s in traps:
    return -5
  if s in treasure:
    return 3
  return max([q_table.getValue(s, a) for a in actions if isValidState(getSPrime(s, a))])

def expectedValue(s, a, q_table):
  return sum([value(sPrime, q_table) * p for (sPrime, p) in transition(s, a).items()]) 

def plotQ(q_table):
  values = [[value((y,x), q_table) for x in range(rows)] for y in range (cols)]
  minValue = min([min(i) for i in values])
  maxValue = max([max(i) for i in values])
  #for w in walls:
    #values[w[0]][w[1]] = minValue # 1, 0????
  values = [[np.interp(values[x][y], [minValue, maxValue], [0, 1]) for y in range(rows)] for x in range(cols)]

  bestActions = []
  for x in range(cols):
    for y in range(rows):
      s = (x, y)
      bestA = 0
      for a in range(len(actions)):
        if isValidState(getSPrime(s, actions[a])) and q_table.getValue(s, actions[a]) > q_table.getValue(s, actions[bestA]):
          bestA = a
      bestActions.append(bestA)
  
  horiz = np.array([actions[i][1] for i in bestActions])
  vert = np.array([-1 * actions[i][0] for i in bestActions])
  
  X = np.arange(0, rows, 1)
  Y = np.arange(0, cols, 1)

  X, Y = np.meshgrid(X, Y)

  _ , ax = plt.subplots()
  ax.quiver(X, Y, horiz, vert)

  plt.imshow(values)
  plt.colorbar()
  plt.show()

def randomValue(q_table):
  return value(random.choice(getAllValidStates()), q_table) if getAllValidStates() else None

def value_iteration(q_table, eps=0.01, max_iter=10000):
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
            new_q_table.setValue(s,a,reward(s) + discount_factor * expectedValue(s,a, q_table))
            delta = max(delta, abs(new_q_table.getValue(s,a) - q_table.getValue(s,a)))
    print("DELTA", delta)
    if delta < eps:
      plotQ(new_q_table)
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
traps = [(6, 5), (1, 1)]  
treasure = [(5, 5)]
# traps = traps + [(6, 7)]
walls = [(4, x) for x in range(8)]
#walls = wallGenerator()

board = setUpEnvironment(traps, treasure, walls, rows, cols)
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