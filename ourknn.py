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
data = dataset.values
rows = np.unique(data[:,0])
cols = np.unique(data[:,1])


utility_matrix = np.zeros((len(rows),len(cols)),dtype=data.dtype)

movie_id_to_umatrix = {}
umatrix_id_to_movie = {}

user_id_to_umatrix = {}
umatrix_id_to_user = {}



r = 0
for user_id in rows:
    c = 0
    for movie_id in cols: 
        umatrix_id_to_movie[c] = movie_id
        movie_id_to_umatrix[movie_id] = c
        c+=1
    user_id_to_umatrix[user_id] = r
    umatrix_id_to_user[r] = user_id
    r+=1




for index, row in traning_data.iterrows():
    user_id = row['userId']
    movie_id = row['movieId']
    rating = row['rating']
    
    row = user_id_to_umatrix[user_id]
    col = movie_id_to_umatrix[movie_id]

    utility_matrix[row][col] = rating

num = 0
i = 0
for x in utility_matrix[user_id_to_umatrix[2],:]:
    if x != 0:
        num += 1
        print(x," ",umatrix_id_to_movie[i])
    i+=1
print(num)

#another sanity check
#movieId,film_index = np.unique(user_rating_test.values[:,1],return_inverse=True)
for f in user_rating_test['movieId']:
    print(utility_matrix[user_id_to_umatrix[user],movie_id_to_umatrix[f]])
    assert utility_matrix[user_id_to_umatrix[user],movie_id_to_umatrix[f]] == 0, "Test data was not removed correctly"

input("HEJ")
print("The size of the matrix is; rows: ",len(rows)," cols: ",len(cols),"\n")


#Finding the k-nearest neighbours of user
result = knn(user_id_to_umatrix[user],utility_matrix)
print("knn for user",user," is")
print("Similar users")
print(result[0],"\n")
print("Unsimilar users")
print(result[1],"\n")
print("Maximum sim",result[2])
print("Minimum sim",result[3])
print("###################################### \n\n\n")

#Make predictions for all the test movies

film_index = np.unique(user_rating_test.values[:,1])
print(film_index)
points = [0] * len(film_index)
for neighbor in result[0]:
    i = 0
    for film in film_index:
        film_id = movie_id_to_umatrix[film]


        rating = utility_matrix[neighbor[0]][film_id]
        if rating != 0:
            if rating > 3:
                points[i] = points[i] + 1 
            else:
                points[i] = points[i] -1
        i+=1

for neighbor in result[1]:
    i = 0
    for film in film_index:
        film_id = movie_id_to_umatrix[film]
        rating = utility_matrix[neighbor[0]][film_id]
        if rating != 0:
            if rating > 3:
                points[i] = points[i] - 0.5 
            else: 
                points[i] = points[i] + 0.5
        i+=1
print(points)
print(list(zip(points,film_index)))
print("Correct ratings\n",user_rating_test)












