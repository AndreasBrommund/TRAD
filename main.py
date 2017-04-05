import pandas
from random import sample
import numpy as np
from sys import argv
from neighbourhood import knn
from prediction import make_prediction
from matrix import build_matrix
from operator import itemgetter
from sklearn.model_selection import train_test_split

def main():
    url_to_data_file = argv[1]
    dataset = read_csv(url_to_data_file)
    
    users = generate_user_sample(671,84)
    
    #TODO Fix later
    user = users[0]

    #TODO add loop
    traning, test = generate_traning_and_test(user,dataset) 

    #Get all umatrix and all the mapings
    umatrix,movie_to_matrix,matrix_to_movie,user_to_matrix,matrix_to_user = build_matrix(dataset,traning)
    
    #Get knn
    sim_users,unsim_users,max_sim,min_sim = knn(user_to_matrix[user],umatrix)

    #Sanity check
    for f in test['movieId']:
        assert umatrix[user_to_matrix[user],movie_to_matrix[f]] == 0, "Test data was not removed correctly"


    print("Num of sim users: ",len(sim_users)," Num of unsim users: ",len(unsim_users))
    print("Max: ",max_sim," Min: ",min_sim)

def read_csv(path):
    return pandas.read_csv(path,
            usecols=['userId','movieId','rating'],
            dtype={'rating':np.float64})

def generate_user_sample(population_size,sample_size):
    return sample(range(1,population_size),sample_size)

def generate_traning_and_test(user,dataset):
    ratings_for_selected_user = dataset[(dataset.userId == user)]
    
    user_rating_train,user_rating_test = train_test_split(
            ratings_for_selected_user,
            train_size = 0.7,test_size = 0.3) 

    #removing the test data from the overall data
    traning_data = dataset[~dataset.isin(user_rating_test).all(1)]

    return traning_data,user_rating_test
    
    






if __name__ == "__main__": main()
