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
#read which user and what film to predict rating for.
user = int(argv[2])
#parsing the csv file with pandas
data_file =  url_to_data_file
dataset = pandas.read_csv(data_file,
        usecols=['userId','movieId','rating'],
        dtype={'rating':np.float64})


#split the dataset into traning and test set
ratings_for_selected_user = dataset[(dataset.userId == user)]
user_rating_train,user_rating_test = train_test_split(ratings_for_selected_user,
        test_size = 0.3, train_size = 0.7) 

#removing the test data from the overall data
traning_data = dataset[~dataset.isin(user_rating_test).all(1)]

print(len(traning_data),"of",len(dataset))

#sanity check..
user_really_gone = traning_data[(traning_data.userId == user)]
removed_test_data = user_really_gone == user_rating_train.sort_index()
print(user_really_gone)
print(user_rating_train.sort_index())
print(removed_test_data)

input("Press [ENTER] to continue...")
print("###################################### \n\n\n")


#Building our utility matrix
data = traning_data.values
rows,row_pos = np.unique(data[:,0],return_inverse=True)
cols,col_pos = np.unique(data[:,1],return_inverse=True)

utility_matrix = np.zeros((len(rows),len(cols)),dtype=data.dtype)
utility_matrix[row_pos,col_pos] = data[:,2]

#another sanity check
movieId,film_index = np.unique(user_rating_test.values[:,1],return_inverse=True)
for f in film_index:
    assert utility_matrix[user,f] == 0, "Test data was not removed correctly"


print("The size of the matrix is; rows: ",len(rows)," cols: ",len(cols),"\n")


#Finding the k-nearest neighbours of user
result = knn(user,utility_matrix,671)
print("knn for user",user," is")
#print(result[0],"\n")
print("Maximum sim",result[1])
print("Minimum sim",result[2])
print("###################################### \n\n\n")

pred = []
#Make predictions for all the test movies
for film in film_index:
    neighbours_who_watched_film = neighbours_that_rated(utility_matrix,result[0],film)
    print(len(neighbours_who_watched_film), " out of ", len(result[0]), "have rated the film")
    prediction = make_prediction(utility_matrix,neighbours_who_watched_film,film)
    print ("Prediction for film ",film, "for user",user,"is",prediction)
    pred.append((film,prediction))
    print("###################################### \n\n\n")


print(list(zip(film_index,movieId)))
print("Correct ratings\n",user_rating_test)
print("Predicted ratings \n",pred)












