
# def initEnvironment(environment):
#   for i in range(10):
#     arr = []
#     for z in range(10):
#       arr.append('')
#     environment.append(arr)
    
# initEnvironment()


# default = []
# initenvironment(default)

def getSPrime(s, a):
  return (s[0] + a[0], s[1] + a[1])

#arrays of tuples signifying the positions
def setUpEnvironment(agentPos, trapsPos, rewardPos, wallsPos, rows, cols):
  board = {s:'' for i in range(rows * cols)}
  for a in agent:
    board[a] = 'a'
  for t in trapsPos:
    board[t] = 't'
  for r in rewardPos:
    board[r] = 'r'
  for w in wallsPos:
    board[w] = 'w' 
  
  board[-1, -1] = (rows, cols)
  return board

board = setUpEnvironment()

def reward(s, a, rewards={'': -1, 'w': -1, 't': -100, 'r': 0}):
  return rewards[board[getSPrime(s, a)]]
    
def createTransition(probs, actions):
  transition = {}
  for i in range(len(probs)):
    transition[actions[i]] = probs[i]
  return transition

transitionDict = createTransition([.25, .25, .25, .25], [(1, 0), (0, 1), (0, -1), (0, 1)])

def transition(a):
  return transitionDict[a]

mdp = mdp(board, actions, transition, reward)


#self, agent, board, actions, transition, reward

def value_iteration(mdp, q_table, eps=0.01, max_iter=1000):
  
s
  def expectedValue():

  for i in range(max_iter):
    new_q_table = q_table.copy()
    delta = 0
    for s in mdp.states:
      for a in mdp.actions:
        new_q_table.setValue(s,a,mdp.reward(s,a) + mdp.discount_factor * mdp.transition(s,a)) #do soemthign
        delta = max(delta, abs(new_q_table.get(s,a) - q_table.get(s,a)))
    if delta < eps:
      return new_q_table
    q_table = new_q_table
  return q_table
    

  