from operator import itemgetter
from similarity import pearson


"""knn finds the k-nearest neighbours of the user
supplied. 

Parameters:
    user := the current user we are building the knn list for
    utility_matrix := the matrix with the realtion for user/items
    k := the number of nearest neighbours returned
    default is 5

Returns:
    Return a tuple with:
        knn list
        max similarity value
        min similarity value 

"""
def knn(user,utility_matrix):
    
    user_ratings = utility_matrix[user,:]

    similar_neighbours = []
    unsimilar_neighbours = []
    max_sim = -100
    min_sim = 100

    #find similarity to all other users
    for i in range(len(utility_matrix)):
        #skip if current user
        if i == user:
            continue

        other_user_ratings = utility_matrix[i,:]
        sim = pearson(user_ratings,other_user_ratings)
        
        min_sim = min(min_sim,sim[0])
        max_sim = max(max_sim,sim[0])
        
        if sim[0] > 0.3:
            similar_neighbours.append((i,sim[0]))
        if sim[0] < -0.3:
            unsimilar_neighbours.append((i,sim[0]))

    #Optimization as per 2nd answer: 
    #http://stackoverflow.com/questions/10695139/sort-a-list-of-tuples-by-2nd-item-integer-value


    return  (similar_neighbours,unsimilar_neighbours,max_sim,min_sim)

"""Find the neighbours that have watched and rated this film"""
def neighbours_that_rated(utility_matrix,neighbours,film):
    result = []
    for neighbour in neighbours:
        index = neighbour[0]
        if utility_matrix[index,film] > 0.0:
            print("Rating",utility_matrix[index,film])
            result.append(neighbour)
    return result






