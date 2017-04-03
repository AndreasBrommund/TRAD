from scipy.stats.mstats import pearsonr

"""similarity(p1,p2) calculates the similarity between two
rows from the utility matrix.

    p1 = user 1 ratings from utility matrix
    p2 = user 2 ratings from utility matrix

    Will return a list which at index 0
    has a decimal value between -1 and 1
    that describes the similartiy between these rows.
"""
def pearson(user1_ratings,user2_ratings):
    #lists with the ratings on films that both user1 and user2 have
    #rated
    user1_rated = []
    user2_rated = []
    
    #will filter out ratings that both have not rated or
    #any version of user1 has rated this film but not user2
    for i in range(len(user1_ratings)):
        if user1_ratings[i] != 0.0 and user2_ratings[i] != 0.0:
            user1_rated.append(user1_ratings[i])
            user2_rated.append(user2_ratings[i])
    #using scipy pearson correlation coefficent to calculate similarty
    sim = pearsonr(user1_ratings,user2_ratings)
    return sim
