import numpy as np
import Tkinter
import matplotlib.pyplot as plt

#this is testBranch
#test 2
x = np.arange(10)
y1 = 2*x
y2 = x

plt.figure(0)
plt.plot(x, y1, 'bo', label = 'Training Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


plt.figure(1)
plt.plot(x, y2, 'bo', label = 'Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
