

"""make prediction
predictions the rating for a given film based
on neighbours using k-nearest neighbour classifiction"""
def make_prediction(utility_matrix,neighbours,film):
    ratings = [0]*11
    for neighbour in neighbours:
        index = neighbour[0]
        
        rating = utility_matrix[index,film]
        weighted_rating = rating * neighbour[1]
        
        result_index = rating_to_index(rating)
        ratings[result_index] += weighted_rating

    #find the rating the was classified, given the highest vote
    prediction = max(ratings)
    return index_to_rating(ratings.index(prediction))

"""Maps from array index to rating
and vice versa"""
def rating_to_index(x):
    return {0.0:0,
            0.5:1,
            1.0:2,
            1.5:3,
            2.0:4,
            2.5:5,
            3.0:6,
            3.5:7,
            4.0:8,
            4.5:9,
            5.0:10
    }[x]

def index_to_rating(x):
    return {0:0.0,
            1:0.5,
            2:1.0,
            3:1.5,
            4:2.0,
            5:2.5,
            6:3.0,
            7:3.5,
            8:4.0,
            9:4.5,
            10:5.0
    }[x]

