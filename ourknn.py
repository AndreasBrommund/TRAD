import pandas
import numpy as np
from scipy.stats.mstats import pearsonr
from sys import argv
from neighbourhood import knn
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


#make prediction for film 1 for user 1
def neighbours_that_rated(neighbours,film):
    result = []
    for neighbour in neighbours:
        index = neighbour[0]
        print("Rating",utility_matrix[index,film])
        if utility_matrix[index,film] > 0.0:
            result.append(neighbour)
    return result
    

neighbours_who_watched_film = neighbours_that_rated(result[0],film)
print(neighbours_who_watched_film)

def rating_index(x):
    return {0.0:0,
            0.5:1,
            1.0:2,
            1.5:3,
            2.0:4,
            2.5:5,
            3.0:6,
            3.5:7,
            4.0:8,
            4.5:9,
            5.0:10
    }[x]

def index_to_rating(x):
    return {0:0.0,
            1:0.5,
            2:1.0,
            3:1.5,
            4:2.0,
            5:2.5,
            6:3.0,
            7:3.5,
            8:4.0,
            9:4.5,
            10:5.0
    }[x]

def make_prediction(neighbours,film):
    ratings = [0]*11
    for neighbour in neighbours:
        index = neighbour[0]
        
        rating = utility_matrix[index,film]
        weighted_rating = rating * neighbour[1]
        
        rate_index = rating_index(rating)
        ratings[rate_index] += weighted_rating

    prediction = max(ratings)


    return index_to_rating(ratings.index(prediction))

prediction = make_prediction(neighbours_who_watched_film,0)
print ("Prediction for film ",film, "for user",user,"is",prediction)























