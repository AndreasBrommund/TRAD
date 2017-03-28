import pandas
import numpy as np
from scipy.stats.mstats import pearsonr

data_file = "100kratings.csv" 
dataset = pandas.read_csv(data_file,
        usecols=['userId','movieId','rating'],
        dtype={'rating':np.float64})

data = dataset.values

#Building our utility matrix
rows,row_pos = np.unique(data[:,0],return_inverse=True)
cols,col_pos = np.unique(data[:,1],return_inverse=True)

utility_matrix = np.zeros((len(rows),len(cols)),dtype=data.dtype)
utility_matrix[row_pos,col_pos] = data[:,2]


row1 = utility_matrix[0,:]
row2 = utility_matrix[1,:]

print(row1)
print(row2)
res1 = []
res2 = []

for i in range(len(row1)):
    if row1[i] != 0.0 and row2[i] != 0.0:
        res1.append(row1[i])
        res2.append(row2[i])

print(res1)
print(res2)


sim = pearsonr(res2,res1)
print("Similarity",sim[0])
