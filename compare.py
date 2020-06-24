from board import Board
from sequential import Sequential
from qlearning import GridWorldAgent
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def comparePlot(sequential, qlearning, recolor=0.98, arrows=True):
  seqValues = sequential.q_table.getValueTable()
  seqValues = sequential.board.recolorAndRescale(seqValues, recolor)
  qValues = qlearning.q_table.getValueTable()
  qValues = qlearning.env.board.recolorAndRescale(qValues, recolor)
  
  _, axr = plt.subplots(1, 2)

  if arrows:
    sequential.board.plotArrows(axr[0], sequential.actions, sequential.board.getBestActions(sequential.q_table, sequential.actions))
    qlearning.env.board.plotArrows(axr[1], qlearning.actions, qlearning.env.board.getBestActions(qlearning.q_table, qlearning.actions))
  axr[0].set_title("Sequential")
  axr[0].imshow(seqValues)
  axr[1].set_title("Q Learning")
  axr[1].imshow(qValues)
  plt.show()

if __name__ == "__main__":
  actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
  board = Board(rand=True)
  sequential = Sequential(actions=actions, board=board, plot=False)
  sequential.value_iteration()
  qlearning = GridWorldAgent(actions=actions, board=board)
  qlearning.run(qlearning.epsLinear, qlearning.epsLinear)
  comparePlot(sequential, qlearning)
