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
from sklearn.ensemble import RandomForestClassifier

def main():
    url_to_data_file = argv[1]
    url_to_film_file = argv[2]
    sample_size = int(argv[3])

    #read only the films genre
    film_dataset = pandas.read_csv(url_to_film_file,sep = "|",usecols=range(5,24))
    

    dataset = read_csv(url_to_data_file)
    population_size = len(np.unique(dataset.values[:,0]))
    users = generate_user_sample(population_size,sample_size)
    
    list_precisions_collaborative = []
    list_recalls_collaborative = []
    total_hits_collaborative = 0
    total_recommendations_collaborative = 0
    total_good_films_collaborative = 0
    

    list_precisions_content = []
    list_recalls_content = []
    total_hits_content = 0
    total_recommendations_content = 0
    total_good_films_content = 0


    #Get all umatrix and all the mapings
    print("Start: build umatrix", file=sys.stderr) 
    umatrix,movie_to_matrix,matrix_to_movie,user_to_matrix,matrix_to_user = matrix.build_matrix(dataset)
    print("End: build umatrix", file=sys.stderr)

    for i,user in enumerate(users):
 
        print("User ",i, " of ", len(users),file=sys.stderr)

        training, test = generate_traning_and_test(user,dataset) 
        matrix.remove(test,umatrix,user_to_matrix,movie_to_matrix)
        
        #Sanity check
        for f in test['movieId']:
            assert umatrix[user_to_matrix[user],movie_to_matrix[f]] == 0, "Test data was not removed correctly"

       
        #Vars
        predicted_films_collaborative = []
        predicted_films_content = []
        
        #Collaborative part
        precision,recall,hits,recommended_films,good_films,predicted_films_collaborative = collaborative(user,umatrix,movie_to_matrix,user_to_matrix,training,test)
        
        list_precisions_collaborative.append(precision)
        list_recalls_collaborative.append(recall)
        total_hits_collaborative += hits
        total_recommendations_collaborative += recommended_films
        total_good_films_collaborative += good_films

        #Content part

        precision,recall,hits,recommended_films,good_films,predicted_films_content = content_predict(user,training,test,film_dataset,3,200)
                    
        list_precisions_content.append(precision)
        list_recalls_content.append(recall)
        total_hits_content += hits
        total_recommendations_content += recommended_films
        total_good_films_content += good_films

        #Hybrid part
        print(predicted_films_collaborative,"\n")
        print(predicted_films_content,"\n")
        print(min(good_films,5))
        print(hybrid(predicted_films_collaborative,predicted_films_content,min(good_films,5)))           
        
        

        #Clean 
        matrix.add(test,umatrix,user_to_matrix,movie_to_matrix)


    if False:
        print("Raw data collaborative\n")
        print("Precisions\n",list_precisions_collaborative)
        print("Recalls\n",list_recalls_collaborative)

        print("Raw data content\n")
        print("Precisions\n",list_precisions_content)
        print("Recalls\n",list_recalls_content)

        print("\nResult collaborative:")
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

        print("\nResult content:")
        print("Total hits: ",total_hits_content)
        print("Total recommendations: ",total_recommendations_content)
        print("Total good films: ",total_good_films_content)

        print("\nPredictions")
        print("Mean: ",np.mean(list_precisions_content))
        print("Median: ",np.median(list_precisions_content))
        print("Overall: ",total_hits_content/total_recommendations_content)
            
        print("\nRecall")
        print("Mean: ",np.mean(list_recalls_content))
        print("Median: ",np.median(list_recalls_content))
        print("Overall: ",total_hits_content/total_good_films_content)

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
        return 0,0,0,0,len(test[(test.rating > 3)]),predictions
        
    precision,recall,hits,recommended_films,good_films = calculate_hit_rate(test,predictions)
    return precision,recall,hits,recommended_films,good_films,predictions 

#Return: Precision, Recal, hits, recommended_films, good_films
def content_predict(user,training,test,film_dataset,depth,trees): 

    film_ratings_training = []
    film_atribute_training = []
    for index, row in training[(training.userId == user)].iterrows():
        movie_id = int(row['movieId'])-1
        rating = row['rating'] 
        if rating > 0:
            film_ratings_training.append(rating)
            film_atribute_training.append(film_dataset.values[movie_id])        

    film_id_test = []
    film_atribute_test = []
    for index, row in test[(test.userId == user)].iterrows():
        movie_id = int(row['movieId'])-1
        rating = row['rating'] 
        if rating > 0:
            film_atribute_test.append(film_dataset.values[movie_id])  
            film_id_test.append(movie_id+1)

    rf = RandomForestClassifier(max_depth = depth,n_estimators = trees)
    rf.fit(film_atribute_training,film_ratings_training)
        
    ratings = rf.predict(film_atribute_test)
 
    res = []

    for index,value in enumerate(ratings):
        if value > 3:
            res.append((int(film_id_test[index]),1.0))

    if len(res) == 0:
        return 0,0,0,0,len(test[(test.rating > 3)]),res

    precision,recall,hits,recommended_films,good_films = calculate_hit_rate(test,res)
    return precision,recall,hits,recommended_films,good_films,res


def hybrid(predicted_films_collaborative,predicted_films_content,films_to_predict):
    predicted_films_hybrid = []

    for index,value in enumerate(predicted_films_collaborative):
        film_id = value[0]
        film_ratio = value[1]
        
        for i,v in enumerate(predicted_films_content):
            
            if(v[0] == film_id):
                if len(predicted_films_hybrid) < films_to_predict:
                    predicted_films_hybrid.append(value)
    
    
    num = 0
    for i in predicted_films_collaborative:
        should_contain = False

        for j in predicted_films_content:
            if i[0] == j[0]:
                should_contain = True
                       
        if should_contain:
            num += 1

        if should_contain and num <= films_to_predict:
            assert i in predicted_films_hybrid , "Not in hybrid list"
        else:
            assert not (i in predicted_films_hybrid), "In hybrid list"


    #Remove films 
    for v in predicted_films_hybrid:
        predicted_films_collaborative.remove(v)
        predicted_films_content.remove((v[0],1))

    #Sanity check
    for i in predicted_films_hybrid:
        for j in predicted_films_collaborative:
            assert i[0] != j[0], "Not removed colllabrative"

    
    for i in predicted_films_hybrid:
        for j in predicted_films_content:
            assert i[0] != j[0], "Not removed conetent"
    
    
    for v in predicted_films_collaborative:
        if v[1] >= 0.8 and len(predicted_films_hybrid) < films_to_predict:
            predicted_films_hybrid.append(v)

    #Sanity check
    assert len(predicted_films_hybrid) <= films_to_predict, "To many films" 


    assert len(set(predicted_films_hybrid)) == len(predicted_films_hybrid), "Have duplicates"

    return predicted_films_hybrid


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
