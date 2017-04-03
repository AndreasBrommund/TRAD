import pandas
import numpy as np
from similarity import pearson
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

#testing with the first two user rows.
row1 = utility_matrix[0,:]
row2 = utility_matrix[1,:]

sim = pearson(row1,row2)

print("The similarity is:",sim[0])






