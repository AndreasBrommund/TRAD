import pandas
from random import sample
import numpy as np
from sys import argv
from neighbourhood import knn
from prediction import predict
import matrix 
from operator import itemgetter
from sklearn.model_selection import train_test_split
import sys

def main():
    url_to_data_file = argv[1]
    sample_size = int(argv[2])

    dataset = read_csv(url_to_data_file)
    population_size = len(np.unique(dataset.values[:,0]))
    users = generate_user_sample(population_size,sample_size)
    
    list_precisions = []
    list_recalls = []
    total_hits = 0
    total_recommendations = 0
    total_good_films = 0
     
    #Get all umatrix and all the mapings
    print("Start", file=sys.stderr) 
    umatrix,movie_to_matrix,matrix_to_movie,user_to_matrix,matrix_to_user = matrix.build_matrix(dataset)
    print("End", file=sys.stderr)

    for i,user in enumerate(users):


        
        print("User ",i, " of ", len(users),file=sys.stderr)

        traning, test = generate_traning_and_test(user,dataset) 

        matrix.remove(test,umatrix,user_to_matrix,movie_to_matrix)
        
        #Get knn
        sim_users,unsim_users,max_sim,min_sim = knn(user_to_matrix[user],umatrix)
        
        #Sanity check
        for f in test['movieId']:
            assert umatrix[user_to_matrix[user],movie_to_matrix[f]] == 0, "Test data was not removed correctly"

        print("Num of sim users: ",len(sim_users)," Num of unsim users: ",len(unsim_users),file=sys.stderr)
        print("Max: ",max_sim," Min: ",min_sim,file=sys.stderr)

        predictions = predict(test,umatrix,movie_to_matrix,sim_users,unsim_users)

        if len(predictions) == 0:
            list_precisions.append(0)
            list_recalls.append(0)
            total_good_films += len(test[(test.rating > 3)])
            matrix.add(test,umatrix,user_to_matrix,movie_to_matrix)

            continue

        film_recommendation = recommendations(test,predictions)
        
        if len(film_recommendation) == 0:
            list_precisions.append(0)
            list_recalls.append(0)
            total_good_films += len(test[(test.rating > 3)])
            
            matrix.add(test,umatrix,user_to_matrix,movie_to_matrix)

            continue

        precision,recall,hits,recommended_films,good_films = calculate_hit_rate(test,film_recommendation) 

        list_precisions.append(precision)
        list_recalls.append(recall)
        total_hits += hits
        total_recommendations += recommended_films
        total_good_films += good_films

        
        matrix.add(test,umatrix,user_to_matrix,movie_to_matrix)

    print("Precisions\n",list_precisions)
    print("Recalls\n",list_recalls)
    print("Total hits: ",total_hits)
    print("Total recommendations: ",total_recommendations)
    print("Total good films: ",total_good_films)

    print("\nPredictions")
    print("Mean: ",np.mean(list_precisions))
    print("Median: ",np.median(list_precisions))
    print("Overall: ",total_hits/total_recommendations)
        
    print("\nRecall")
    print("Mean: ",np.mean(list_recalls))
    print("Median: ",np.median(list_recalls))
    print("Overall: ",total_hits/total_good_films)

    print("\nPopulation size: ",population_size)
    print("Sample size: ",sample_size)


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
    """good_films = test[(test.rating > 3)]

    films_to_recomend = []
    if  len(good_films) < 5: 
        films_to_recommend = predictions[:len(good_films) ] 
    else:   
        films_to_recommend = predictions[:5]
    """

    return predictions

def calculate_hit_rate(test,films_to_recommend):
    good_films = test[(test.rating > 3)]
    hits = 0

    for rating in films_to_recommend:
        hit = good_films[(good_films.movieId == rating[0])]

        if len(hit) == 1:
            hits += 1

    precision = 0
    recall = 0
    
    if len(films_to_recommend) != 0:
        precision = hits/len(films_to_recommend)


    if len(good_films) != 0:
        recall = hits/len(good_films)
    
    return precision,recall,hits,len(films_to_recommend),len(good_films)



if __name__ == "__main__": main()
