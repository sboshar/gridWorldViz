from tabularq import TabularQ
import matplotlib.pyplot as plt
import numpy as np
import random

#Global
###############################################
actions =  [(-1, 0), (1, 0), (0, -1), (0, 1), (1,1)]
rows = 10
cols =  10
traps = [(3, x) for x in range(3, 6)]
traps += [(5, x) for x in range(3, 6)]
traps += [(x, 3) for x in range(3, 6)]
traps += [(x, 5) for x in range(3, 6)]

walls = []
treasure = [(4, 4)]
discount_factor = .95
trapReward = -5
treasureReward = 3
pTransition = 0.7
################################################

def getSPrime(s, a):
  """ Returns the state you would travel to taking action a from state s

  Args:
      s (tuple): state
      a (tuple): action

  Returns:
      tuple: new state, s'
  """
  return (s[0] + a[0], s[1] + a[1])
  
def wallGenerator(availablePos, minLength, maxLength, minNum, maxNum):
  """Generates a random number of groups of wals between minNum and maxNum
  and each of the groups of walls has a length between minLength and maxLength

  Args:
      availablePos (array): array of tuples of available positions
      minLength (int): min length of wall group 
      maxLength (int): max length of a wall group
      minNum (int): min number of wall groups
      maxNum (int): max number of wall groups

  Returns:
      array: array of positions of walls
  """
  newWalls = []
  #numbers of wall sequences
  numWallGroup = random.randint(minNum, maxNum)
  directions = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])

  for i in range(numWallGroup):
    #first pos in the walls sequence
    curr = random.choice(availablePos)
    availablePos.remove(curr)

    length = random.randint(minLength, maxLength) 
    #generate each random wall sequence
    for i in range(length):
      np.random.shuffle(directions)
      isCornered = True
      #iterates of directions and if finds a vlid direction it sets
      #isCornered to False and breaks
      for i in range(len(directions)):
        temp = getSPrime(curr, tuple(directions[i]))
        if isValidState(temp) and temp not in newWalls:
          isCornered = False
          break
      #if no valide directions exist
      if isCornered:
        break
      #if isCornered = false, add the next position to newWalls
      # and remove it from avialable walls
      if temp:
        newWalls += [temp]
        if (temp in availablePos):
          availablePos.remove(temp)
        curr = temp
  return newWalls

def itemGenerator(availablePos, minItems, maxItems):
  """Used to generate both positions of traps and rewards. Generates
  between minItems and maxItems of the reward/trap and gives them a 
  random avialable position.

  Args:
      availablePos (array): array of tuples of avialable positions
      minItems (int): min number items
      maxItems (int): max number item

  Returns:
      array: array of postions of items
  """
  #inclusive on both ends
  numRewards = random.randint(minItems, maxItems)
  newRewards = []
  for i in range(numRewards):
    pos = random.choice(availablePos)
    newRewards.append(pos)
    availablePos.remove(pos)
  return newRewards    

def generateBoard(minWallLength, maxWallLength, minNumWalls, maxNumWalls, minR, maxR, minT, maxT):
  """This function used the other generator functions in order to generate an entire board
  such that the items on the board do not conflict in location. Returns arrays of walls, rewards, traps

  Args:
      minWallLength (int): minimum number of walls in each wall group
      maxWallLength (int): maximum number of walls in each wall group
      minNumWalls (int): minimum number of wall groups
      maxNumWalls (int): maximum number of wall groups
      minR (int): minimum number of rewards
      maxR (int): maximum number of rewards
      minT (int): minimum number of traps
      maxT (int): maximum number of traps

  Returns:
      (array, array, array): an array of wall positions, an array of reward positions, and an
                             array of trap positions
  """
  availablePos = [(x, y) for y in range(cols) for x in range(rows)]
  walls = wallGenerator(availablePos, minWallLength, maxWallLength, minNumWalls, maxNumWalls)
  rewards = itemGenerator(availablePos, minR, maxR)
  traps = itemGenerator(availablePos, minT, maxT)
  return (walls, rewards, traps)
  
#arrays of tuples signifying the positions
def setUpEnvironment(randomEnv=False, printEnv=False):
  """Takes arrays of tuples containing the locations of the board 
      elements includuing traps, reward etc, and creates a board of dim row by cols
      with those features.

  Args:
      randomEnv (bool, optional): If true generates random environment. 
                                  Defaults to False.
      printEnv (bool, optional): If true prints array representation of environment. 
                                 Defaults to False.

  Returns:
      2d np array: the board, which is a 2d np array of characters. ' ' is an empty space, 
                   'w' is a wall, 't' is a trap, and 'r' is a reward. 
  """
  #allows global redefining for random environments
  global walls, treasure, traps
  if randomEnv: 
    walls, treasure, traps = generateBoard(4, 6, 2, 4, 1, 3, 1, 4)
  board = np.array([[' ' for c in range(cols)] for r in range(rows)])
  #sets up board to be 2d numpy array of char
  for w in walls:
    board[w] = 'w' 
  for t in traps:
    board[t] = 't'
  for r in treasure:
    board[r] = 'r'
  if printEnv: 
    print("-------------------------------------------------------------------------")
    print("Board: ")
    print(np.array(board))
    print(f"{len(treasure)} treasure(s), {len(walls)} walls, {len(traps)} trap(s)")
    print("Walls:", walls)
    print("Traps:", traps)
    print("Treasures:", treasure)
    print("-------------------------------------------------------------------------")
  return board

def reward(s, rewards={' ': 0, 'w': None, 't': trapReward, 'r': treasureReward}):
  """ Returns the reward for leaving the current state s
  Args:
      s (tuple): state
      rewards (dict, optional): Dictionary that maps board positions (strings) totheir reward (ints) 
                                Defaults to {' ': 0, 'w': -1, 't': -5, 'r': 10}.

  Returns:
      float: the reward of leaving a state
  """
  return rewards[board[s]]

def transition(s, a):
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
  
  validNeighbors = [getSPrime(s, a) for a in actions if isValidState(getSPrime(s, a), board)]
  if len(validNeighbors) == 1: #Always transition to validNeighbor[0] if there is one valid neighbor
    return {validNeighbors[0] : 1}
  elif len(validNeighbors) == 0:
    return {(0,0) : 0}
  return {sPrime: pTransition if (getSPrime(s, a) == sPrime) else ((1 - pTransition) / (len(validNeighbors) - 1)) for sPrime in validNeighbors}
   
def isValidState(s, b=np.array([[' ' for c in range(cols)] for r in range(rows)])):
  """ Returns True if state s is a valid state within board b, and False if the state is invalid.
      Invalid states are ones with improper indices (out of bounds), or ones that are walls. Valid
      states are all others.

  Args:
      s (tuple): state
      b (2d np array, optional): the state space of traps, walls, rewards, and blank spaces. 
                                 Defaults to np.array([[' ' for c in range(cols)] for r in range(rows)]).

  Returns:
      bool: True if s is a valid state, False otherwise
  """
  
  if (s[0] >= rows or s[0] < 0):
    return False
  if (s[1] >= cols or s[1] < 0):
    return False
  if b[s] == 'w':
    return False
  return True

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
  return 0 #Return 0 if a state has no valid neighbors

def expectedValue(s, a, q_table):
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
  return sum([value(sPrime, q_table) * p for (sPrime, p) in transition(s, a).items()]) 

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

def value_iteration(q_table, eps=0.01, max_iter=10000, plot=False, plotEvery=False, verbose=False, recolor=0.98, arrows=True):
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
  for i in range(max_iter):
    new_q_table = q_table.copy()
    if plotEvery: plotQ(new_q_table, verbose=verbose, recolor=recolor, arrows=arrows)
    delta = 0  
    for x in range(len(board)):
      for y in range(len(board[0])):
        s = (x, y)
        for a in actions:
          if isValidState(s, board):
            #update q value
            new_q_table.setValue(s,a,reward(s) + discount_factor * expectedValue(s,a, q_table))
            #define delta to me the max change in any q value
            delta = max(delta, abs(new_q_table.getValue(s,a) - q_table.getValue(s,a)))
    if verbose: 
      print("-------------------------------------------------------------------------")
      print(f"Iteration {i}: DELTA {delta}")
      print("-------------------------------------------------------------------------")
    #q_table is not changing 
    if delta < eps:
      if plot: plotQ(new_q_table, verbose=verbose, recolor=recolor, arrows=arrows)
      return new_q_table
    q_table = new_q_table.copy()
  return q_table

if __name__ == "__main__":
  board = setUpEnvironment(randomEnv=True, printEnv=True)
  q_table = TabularQ(board, actions)
  _ = value_iteration(q_table, plot=True, plotEvery=False, verbose=False, recolor=1, arrows=True)