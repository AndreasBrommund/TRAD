import pandas
import numpy as np
from similarity import pearson
from scipy.stats.mstats import pearsonr
from sys import argv

#reading url/path from command line argument
url_to_data_file = argv[1] 

#parsing the csv file with pandas
data_file =  url_to_data_file
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






