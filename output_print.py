import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = pd.read_csv(".\output.txt", sep=' ')
y = np.array(f.iloc[:,1],dtype = 'float')
x = np.linspace(0,len(y), len(y))
# y = np.array(f.iloc[:,7],dtype = 'float')
# t = np.array(f.iloc[:,8],dtype = 'float')
# y = y+t/60
# t = np.arange(len(y))
# t2 = t[y<42.1]
# t3 = t[y>42] 
# t4 = np.intersect1d(t2,t3)
#print(t4)
#print(f.iloc[:,8])
plt.figure()
plt.plot(x,y,'b')

plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

