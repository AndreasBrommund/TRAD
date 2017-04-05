import pandas
from random import sample
import numpy as np
from sys import argv
from neighbourhood import knn
from prediction import predict
from matrix import build_matrix
from operator import itemgetter
from sklearn.model_selection import train_test_split

def main():
    url_to_data_file = argv[1]
    sample_size = int(argv[2])

    dataset = read_csv(url_to_data_file)
    population_size = len(np.unique(dataset.values[:,0]))
    print(population_size)
    users = generate_user_sample(population_size,sample_size)
    
    hits = []

    for i,user in enumerate(users):
        print("User ",i, " of ", len(users))

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

        predictions = predict(test,umatrix,movie_to_matrix,sim_users,unsim_users)

        if len(predictions) == 0:
            hits.append(0)
            continue

        film_recommendation = recommendations(test,predictions)
        
        if len(film_recommendation) == 0:
            hits.append(0)
            continue

        hit_rate = calculate_hit_rate(test,film_recommendation) 

        hits.append(hit_rate)

    print(hits)
    print(np.mean(hits))
    print(np.median(hits))
    

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
    
    
def recommendations(test,predictions):
    good_films = test[(test.rating > 3)]

    films_to_recomend = []
    if  len(good_films) < 5: 
        films_to_recommend = predictions[:len(good_films) ] 
    else:   
        films_to_recommend = predictions[:5]

    return films_to_recommend

def calculate_hit_rate(test,films_to_recommend):
    good_films = test[(test.rating > 3)]
    hits = 0

    for rating in films_to_recommend:
        hit = good_films[(good_films.movieId == rating[0])]

        if len(hit) == 1:
            hits += 1
            
    return hits/len(films_to_recommend)

if __name__ == "__main__": main()
