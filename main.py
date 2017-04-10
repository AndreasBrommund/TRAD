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
    
    list_precisions_collaborative = []
    list_recalls_collaborative = []
    total_hits_collaborative = 0
    total_recommendations_collaborative = 0
    total_good_films_collaborative = 0
     
    #Get all umatrix and all the mapings
    print("Start: build umatrix", file=sys.stderr) 
    umatrix,movie_to_matrix,matrix_to_movie,user_to_matrix,matrix_to_user = matrix.build_matrix(dataset)
    print("End: build umatrix", file=sys.stderr)

    for i,user in enumerate(users):
 
        print("User ",i, " of ", len(users),file=sys.stderr)

        traning, test = generate_traning_and_test(user,dataset) 
        matrix.remove(test,umatrix,user_to_matrix,movie_to_matrix)
        
        #Sanity check
        for f in test['movieId']:
            assert umatrix[user_to_matrix[user],movie_to_matrix[f]] == 0, "Test data was not removed correctly"

       

        #Collaborative part
        precision,recall,hits,recommended_films,good_films = collaborative(user,umatrix,movie_to_matrix,user_to_matrix,traning,test)
        
        list_precisions_collaborative.append(precision)
        list_recalls_collaborative.append(recall)
        total_hits_collaborative += hits
        total_recommendations_collaborative += recommended_films
        total_good_films_collaborative += good_films

        #Content part

        #Hybrid part

        
        matrix.add(test,umatrix,user_to_matrix,movie_to_matrix)


    print("Result collaborative:")

    print("Precisions\n",list_precisions_collaborative)
    print("Recalls\n",list_recalls_collaborative)
    print("Total hits: ",total_hits_collaborative)
    print("Total recommendations: ",total_recommendations_collaborative)
    print("Total good films: ",total_good_films_collaborative)

    print("\nPredictions")
    print("Mean: ",np.mean(list_precisions_collaborative))
    print("Median: ",np.median(list_precisions_collaborative))
    print("Overall: ",total_hits_collaborative/total_recommendations_collaborative)
        
    print("\nRecall")
    print("Mean: ",np.mean(list_recalls_collaborative))
    print("Median: ",np.median(list_recalls_collaborative))
    print("Overall: ",total_hits_collaborative/total_good_films_collaborative)

    print("\nPopulation size: ",population_size)
    print("Sample size: ",sample_size)






#Return: Precision, Recal, hits, recommended_films, good_films
def collaborative(user,umatrix,movie_to_matrix,user_to_matrix,traning,test):
    #Get knn
    sim_users,unsim_users,max_sim,min_sim = knn(user_to_matrix[user],umatrix)

    print("Num of sim users: ",len(sim_users)," Num of unsim users: ",len(unsim_users),file=sys.stderr)
    print("Max: ",max_sim," Min: ",min_sim,file=sys.stderr)

    predictions = predict(test,umatrix,movie_to_matrix,sim_users,unsim_users)

    if len(predictions) == 0:
        #Precision, Recal, hits, recommended_films, good_films
        return 0,0,0,0,len(test[(test.rating > 3)])
        
    film_recommendation = recommendations(test,predictions)

    if len(film_recommendation) == 0:
        #Precision, Recal, hits, recommended_films, good_films    
        return 0,0,0,0,len(test[(test.rating > 3)])

    return calculate_hit_rate(test,film_recommendation) 

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
