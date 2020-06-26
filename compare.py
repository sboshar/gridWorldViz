from board import Board
from sequential import Sequential
from qlearning import QLearning
from sarsa import Sarsa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

#want, compare plot, compare grid, accept a list of models
#compare grid
class CompareModels(object):
  def __init__(self, modelClass, num_epochs=10000, actions = [(-1, 0), (1, 0), (0, -1), (0, 1)], board = Board(rand=True),
              recolor=0.98, arrows=True, max_steps=100):
    
    #how to change the eps function form linear!
    self.modelClasses = modelClasses
    self.max_steps = max_steps
    self.actions = actions
    self.board = board
    self.num_epochs = num_epochs
    self.models = []
    self.rewards = []
    self.counts = []

    for i in range(len(modelClasses)):
      if modelClasses[i] == Sequential:
        sequential = modelClasses[i](actions=actions, board=board, plot=False)
        sequential.value_iteration()
        self.rewards.append([])
        self.counts.append([])
        self.models.append(sequential)
      else:
        model = modelClasses[i](actions=actions, board=board, max_steps=max_steps, num_epochs=self.num_epochs)
        (reward, count) = model.run(model.epsilonLogDecay, model.epsilonLogDecay)
        self.rewards.append(reward)
        self.counts.append(count)
        self.models.append(model)

    self.model_names = [str(m) for m in self.models]
    self.recolor = recolor
    self.arrows = arrows

  def average(self, array, amount):
    new = []
    for i in range(len(array)):
      if i % amount:
        new.append(np.mean(array[i: i + amount]))
    return new

  def compareRewards(self, smooth=100):
    num_plots = len(self.rewards)
    if Sequential in self.modelClasses:
      num_plots -= 1

    _, axr = plt.subplots(2, num_plots)
    z = 0 
    for i in range(len(self.rewards)):
      if self.rewards[i]: 
        axr[0, z].set_title(self.model_names[i])
        axr[0, z].plot(self.rewards[i])
        axr[1, z].plot(self.average(self.rewards[i], smooth))
        z += 1
      else:
        z -= 1
    plt.show() 

  def compareGrids(self):
    modelValues = []
    for model in self.models:
      values = model.q_table.getValueTable()
      values = model.board.recolorAndRescale(values, self.recolor)
      modelValues.append(values)

    
    _, axr = plt.subplots(1, len(self.models))

    if self.arrows:
      for i in range(len(self.models)):
        if (type(self.models[i]) == Sequential):
          self.models[i].board.plotArrows(axr[i], self.models[i].actions, self.models[i].board.getBestActions(self.models[i].q_table, self.models[i].actions))
        else:
          self.models[i].env.board.plotArrows(axr[i], self.models[i].actions, self.models[i].env.board.getBestActions(self.models[i].q_table, self.models[i].actions))
      
    for i in range(len(self.models)):
      axr[i].set_title(self.model_names[i])
      axr[i].imshow(modelValues[i])
    plt.show()

  
if __name__ == "__main__":
  modelClasses = [Sequential, QLearning, Sarsa]
  c = CompareModels(modelClasses)
  c.compareGrids()
  c.compareRewards()
