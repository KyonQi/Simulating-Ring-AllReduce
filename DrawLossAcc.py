import matplotlib.pyplot as plt

Iteration = 2000
Show_Iter = 10

def draw_loss(Loss_list, epoch=Iteration / Show_Iter):
  plt.cla()
  x1 = range(1, epoch+1)
  print(x1)
  y1 = Loss_list
  print(y1)
  plt.title('Train loss', fontsize=15)
  plt.plot(x1, y1, '.-')
  plt.xlabel('epoches', fontsize=15)
  plt.ylabel('Train loss', fontsize=15)
  plt.grid()
  #plt.savefig("./lossAndacc/NoAttack_Train_loss.png")
  plt.show()

def draw_acc(accu_list, epoch=Iteration / Show_Iter):
  plt.cla()
  x1 = range(1, epoch+1)
  print(x1)
  y1 = accu_list
  print(y1)
  plt.title('Test Accu', fontsize=15)
  plt.plot(x1, y1, '.-')
  plt.xlabel('epoches', fontsize=15)
  plt.ylabel('Test Accu', fontsize=15)
  plt.grid()
  #plt.savefig("./lossAndacc/NoAttack_Test_acc.png")
  plt.show()