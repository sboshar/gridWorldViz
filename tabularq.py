class TabularQ:

  def __init__(self, states, actions):
    self.states = states
    self.actions = actions
    #create a dictionary for the q q_table intialize q values to 0.0
    self.q_table = {((s,a), 0.0) for s in states for a in actions}
  
  def getQ(self):
    return self.q_table

  def setQ(self, new_q)
    #updates/sets the q table to be new_q expect dict
    self.q_table.update(new_q)
 
  def getValue(self, s, a):
    return self.q_table[(s,a)]

  def setValue(self, s, a, v ):
    self.q_table[(s,a)] = v

  def copy(self):
    copy = TabularQ(self.states, self.actions)
    copy.setQ(self.q_table)
    return copy
  
    
      
  

