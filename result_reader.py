import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import pandas
from sys import argv

path = argv[1]

raw_data = []
lines = []
nr_lines = 0
labels = ["Collaborative","Collaborative","Content","Content","Hybrid"]



with open(path) as f:
    for line in f:
        lines.append(line)
        if "[" in line:
            raw_data.append((nr_lines,list(map((lambda x: float(x)),line.replace("[","").replace("]","").split(",")))))
        nr_lines += 1



i = 0
for l,d in raw_data:
    dataset = pandas.DataFrame(d)
    l -=1 
    w = np.ones_like(dataset.values)/len(dataset.values)
    plt.hist(dataset.values,weights=w)
    plt.title(labels[i]+ " " + lines[l])
    plt.savefig("./img/"+str(i)+"_"+labels[i]+ "_1_" + lines[l])

    dataset.plot(kind='box')
    plt.title(labels[i]+ " "+ lines[l])
    plt.savefig("./img/"+str(i)+"_"+labels[i]+ "_2_" + lines[l])
    plt.show()
    i +=1



