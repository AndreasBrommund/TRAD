import numpy as np

def build_matrix(dataset,traning_data):
    data = dataset.values
    rows = np.unique(data[:,0])
    cols = np.unique(data[:,1])

    #Build a rows x cols matrix with zeros
    utility_matrix = np.zeros((len(rows),len(cols)),dtype=data.dtype)

    #Init mappers for movieId
    movie_id_to_umatrix = {}
    umatrix_id_to_movie = {}

    #Init mappers for userId
    user_id_to_umatrix = {}
    umatrix_id_to_user = {}

    #Calculate the mapping


    row = 0
    for user_id in rows:
        user_id_to_umatrix[user_id] = row
        umatrix_id_to_user[row] = user_id
        row+=1

    col = 0
    for movie_id in cols: 
        umatrix_id_to_movie[col] = movie_id
        movie_id_to_umatrix[movie_id] = col
        col+=1



    #Add values to the utility matrix
    for index, row in traning_data.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        
        row = user_id_to_umatrix[user_id]
        col = movie_id_to_umatrix[movie_id]

        utility_matrix[row][col] = rating
    
    return utility_matrix, movie_id_to_umatrix, umatrix_id_to_movie, user_id_to_umatrix, umatrix_id_to_user


     
     
