import numpy as np 
from operator import itemgetter

#Make predictions for all the test movies
def predict(test,umatrix,movie_to_matrix,sim_users,unsim_users):
    
    film_index = np.unique(test.values[:,1])

    positive_counter = [0] * len(film_index)
    negative_counter = [0] * len(film_index)
    
    #Add point for similar users
    for neighbor in sim_users:
        i = 0
        
        for film in film_index:
            film_id = movie_to_matrix[film]
            rating = umatrix[neighbor[0]][film_id]
            
            if rating != 0:
                if rating > 3:
                    positive_counter[i] = positive_counter[i] + 1 
                else:
                    negative_counter[i] = negative_counter[i] +1
            i+=1

    #Add points for unsimilar users
    for neighbor in unsim_users:
        i = 0
        
        for film in film_index:
            film_id = movie_to_matrix[film]
            rating = umatrix[neighbor[0]][film_id]
            
            if rating != 0:
                if rating > 3:
                    negative_counter[i] =negative_counter[i] + 0.5 
                else: 
                    positive_counter[i] = positive_counter[i] + 0.5
            i+=1

    return calculate_percentage(positive_counter,negative_counter,film_index)

def calculate_percentage(positive_counter,negative_counter,film_index):
    recommendations = []

    for index in range(len(film_index)):
        positive = positive_counter[index]
        negative = negative_counter[index]
        numbers_of_ratings = positive+negative
            
        if numbers_of_ratings > 0:
            percentage = positive/numbers_of_ratings
            #print(percentage)

            if percentage >= 0.5:
                recommendations.append((film_index[index],percentage))
    
    return sorted(recommendations,key=itemgetter(1),reverse=True)
    
