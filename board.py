import numpy as np 
import matplotlib.pyplot as plt
import random

class Board(object):
  def __init__(self, rows=10, cols=10, walls=[], treasures=[], traps=[], trapReward=-5, treasureReward=3, rand=False):
    """ Creates a random board. 
    """
    self.trapReward = trapReward 
    self.treasureReward = treasureReward
    self.rows = rows
    self.cols = cols
    self.shape = (rows, cols)
    self.walls = walls
    self.treasures = treasures
    self.traps = traps
    self.availablePos =[(x, y) for x in range(self.rows) for y in range(self.cols)]
    self.initBoard(rand)

  def wallGenerator(self, minLength, maxLength, minNum, maxNum):
    """Generates a random number of groups of wals between minNum and maxNum
    and each of the groups of walls has a length between minLength and maxLength

    Args:
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
      curr = random.choice(self.availablePos)
      self.availablePos.remove(curr)

      length = random.randint(minLength, maxLength) 
      #generate each random wall sequence
      for i in range(length):
        np.random.shuffle(directions)
        isCornered = True
        #iterates of directions and if finds a vlid direction it sets
        #isCornered to False and breaks
        for i in range(len(directions)):
          temp = self.getSPrime(curr, tuple(directions[i]))
          if self.isValidState(temp) and temp not in newWalls:
            isCornered = False
            break
        #if no valide directions exist
        if isCornered:
          break
        #if isCornered = false, add the next position to newWalls
        # and remove it from avialable walls
        if temp:
          newWalls += [temp]
          if (temp in self.availablePos):
            self.availablePos.remove(temp)
          curr = temp
    return newWalls


  def itemGenerator(self, minItems, maxItems):
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
    for _ in range(numRewards):
      pos = random.choice(self.availablePos)
      newRewards.append(pos)
      self.availablePos.remove(pos)
    return newRewards    
    

  def generateBoard(self, minWallLength, maxWallLength, minNumWalls, maxNumWalls, minR, maxR, minT, maxT):
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
    #availablePos = [(x, y) for y in range(self.cols) for x in range(self.rows)]
    walls = self.wallGenerator(minWallLength, maxWallLength, minNumWalls, maxNumWalls)
    rewards = self.itemGenerator(minR, maxR)
    traps = self.itemGenerator(minT, maxT)
    return (walls, rewards, traps)
  
#arrays of tuples signifying the positions
  def initBoard(self, rand):  
    self.board = np.array([[' ' for c in range(self.cols)] for r in range(self.rows)])
    if rand:
      self.walls, self.treasures, self.traps = self.generateBoard(4, 6, 2, 4, 1, 3, 1, 4)
    #sets up board to be 2d numpy array of char
    for w in self.walls:
      self.board[w] = 'w' 
    for t in self.traps:
      self.board[t] = 't'
    for r in self.treasures:
      self.board[r] = 'r'

    if not rand:
      self.initializeAvailablePos()
      
    #handle avialable pos
    
    # if printEnv: 
    #   print("-------------------------------------------------------------------------")
    #   print("Board: ")
    #   print(np.array(board))
    #   print(f"{len(treasures)} treasures(s), {len(walls)} walls, {len(traps)} trap(s)")
    #   print("Walls:", walls)
    #   print("Traps:", traps)
    #   print("treasuress:", treasures)
    #   print("-------------------------------------------------------------------------")

  def reward(self, s, rewards=None):
    """ Returns the reward for leaving the current state s
    Args:
        s (tuple): state
        rewards (dict, optional): Dictionary that maps board positions (strings) totheir reward (ints) 
                                  Defaults to {' ': 0, 'w': -1, 't': -5, 'r': 10}.

    Returns:
        float: the reward of leaving a state
    """
    if not rewards:
      rewards={' ': -0.1, 'w': None, 't': self.trapReward, 'r': self.treasureReward}
    return rewards[self.board[s]]
    
  def initializeAvailablePos(self):
    self.availablePos = [(x, y) for x in range(self.rows) for y in range(self.cols) if self.board[x][y] == ' ']
  
  def getRandomState(self): 
    return random.choice(self.availablePos)

  def getSPrime(self, s, a):
    """ Returns the state you would travel to taking action a from state s

    Args:
        s (tuple): state
        a (tuple): action

    Returns:
        tuple: new state, s'
    """
    return (s[0] + a[0], s[1] + a[1])

  def isTerminal(self, state):
    return state in self.traps + self.treasures

  def isValidState(self, s):
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
    
    if (s[0] >= self.rows or s[0] < 0):
      return False
    if (s[1] >= self.cols or s[1] < 0):
      return False
    if self.board[s] == 'w':
      return False
    return True
  

  def __str__(self):
    return str(np.where(self.board==' ', '☐', self.board))

  def plotArrows(self, ax, actions, bestActions):
    horiz = np.array([actions[i][1] if i != -1 else 0 for i in bestActions])
    vert = np.array([-1 * actions[i][0] if i != -1 else 0 for i in bestActions]) 
    Y = np.arange(0, self.rows, 1)
    X = np.arange(0, self.cols, 1)
    X, Y = np.meshgrid(X, Y)
    # _ , ax = plt.subplots()
    ax.quiver(X, Y, horiz, vert)

  
  def getBestActions(self, q_table, actions):
    bestActions = []
    for x in range(self.rows):
      for y in range(self.cols):
        s = (x, y)
        bestA = 0
        if not self.isValidState(s):
          bestActions.append(-1)
          continue
        for a in range(len(actions)):
          if self.isValidState(self.getSPrime(s, actions[a])): #find a valid best action, bestA
            bestA = a
            break
        for a in range(len(actions)): #compare each valid action to bestA
          if self.isValidState(self.getSPrime(s, actions[a])) and \
            q_table.getValue(s, actions[a]) >= q_table.getValue(s, actions[bestA]):
            bestA = a
        bestActions.append(bestA)
    return bestActions

  def recolorAndRescale(self, values, recolor):
    #min and max value of values used for caling the color purposes
    maxValue = max([max(i) for i in values])

    newMin = values[self.traps[0][0]][self.traps[0][1]]
    #if you are recoloring, set new min to be slightly higher than the second lowest value
    if recolor and self.traps:
      minValue = 0
      for y in range(self.rows):
        for x in range(self.cols):
          if (not (y, x) in self.traps) and minValue > values[y][x]:
            minValue = values[y][x]

      newMin = minValue + (values[self.traps[0][0]][self.traps[0][1]] - minValue) * (1 - recolor)
      for t in self.traps:
        values[t[0]][t[1]] = newMin
        
    #colors the walls as the darkest color on the plot
    for w in self.walls:
      values[w[0]][w[1]] = newMin
    
    #map to 0-1 range
    values = [[np.interp(values[x][y], [newMin, maxValue], [0, 1]) for y in range(self.cols)] for x in range(self.rows)]
    return values

  def plot(self, q_table, actions, verbose=False, recolor=0.98, arrows=True):
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
    #values = [[value((y,x), q_table) for x in range(cols)] for y in range (rows)]
    values = q_table.getValueTable()

    if verbose: 
      print("-------------------------------------------------------------------------")
      print("Value Array:")
      print(np.round(np.array(values), 3))
      print("-------------------------------------------------------------------------")

    values = self.recolorAndRescale(values, recolor)
    
    #creates an array of shape [1, row*col] where each element represents the best action to take in that state
    if arrows:
      bestActions = self.getBestActions(q_table, actions)
      _ , ax = plt.subplots()
      self.plotArrows(ax, actions, bestActions)
    
    plt.imshow(values)
    plt.colorbar()
    plt.show()
 
if __name__ == "__main__":
  env = Board(rand=True)
  print(env)
