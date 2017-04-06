import numpy as np
import pandas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

from sys import argv


path = argv[1]
dataset = pandas.read_csv(path)


w = np.ones_like(dataset.values)/len(dataset.values)
plt.hist(dataset.values,weights=w)

dataset.plot(kind='box')
plt.show()


from sys import exit
exit(0)
