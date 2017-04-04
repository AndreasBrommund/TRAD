import pandas
import numpy as np
from scipy.stats.mstats import pearsonr
from sys import argv
from neighbourhood import knn
from neighbourhood import neighbours_that_rated
from prediction import make_prediction
from sklearn.model_selection import train_test_split

#reading url/path from command line argument
url_to_data_file = argv[1] 

#parsing the csv file with pandas
data_file =  url_to_data_file
dataset = pandas.read_csv(data_file,
        usecols=['userId','movieId','rating'],
        dtype={'rating':np.float64})

data = dataset.values

#split the dataset into traning and test set

user2 = dataset[(dataset.userId == 2)]

user2_test,user2_train = train_test_split(user2,test_size=0.3,train_size=0.7)


newDataframe = dataset[~dataset.isin(user2_test).all(1)]
user2reallygone = newDataframe[(dataset.userId == 2)]



print(user2reallygone)
print(user2_train.sort())
input("STOP HERE")



#Building our utility matrix
rows,row_pos = np.unique(data[:,0],return_inverse=True)
cols,col_pos = np.unique(data[:,1],return_inverse=True)

utility_matrix = np.zeros((len(rows),len(cols)),dtype=data.dtype)
utility_matrix[row_pos,col_pos] = data[:,2]

print("The size of the matrix is; rows: ",len(rows)," cols: ",len(cols))

user = int(argv[2])
film = int(argv[3])
#Finding the k-nearest neighbours of user 1.
result = knn(user,utility_matrix,30)
print("knn for user",user," is")
print(result[0])
print("Maximum sim",result[1])
print("Minimum sim",result[2])



neighbours_who_watched_film = neighbours_that_rated(utility_matrix,result[0],film)
print(neighbours_who_watched_film)


prediction = make_prediction(utility_matrix,neighbours_who_watched_film,film)
print ("Prediction for film ",film, "for user",user,"is",prediction)























