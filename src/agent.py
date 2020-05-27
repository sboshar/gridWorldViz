class Agent:
  def __init__(self, xpos, ypos):
    self.__x = xpos
    self.__y = ypos
    
  def move(self, action):
    self.__x += action[0]
    self.__y += action[1]
  
  def getX(self):
    return self.__x
  
  def getY(self):
    return self.__y

  def setX(self, x):
    self.__x = x

  def setY(self, y):
    self.__y = y
  
  def __str__(self):
    return f"({self.__x}, {self.__y})"
