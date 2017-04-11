import pandas
import numpy as np
from sys import argv

file_path = argv[1]
film_name_path = argv[2]
old = argv[3]
sep = argv[4]

#source: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
ratings = pandas.read_csv(file_path,sep=sep)
movies = pandas.read_csv(film_name_path,sep='|',usecols=range(5))
df = pandas.merge(ratings,movies)

group = df.groupby(['name','movieId'])['rating'].sum().reset_index()
sort = group.sort_values('rating',ascending=False)
print(sort[:10])

result=[]
for i,v in sort[:5].iterrows():
    result.append((v['movieId'],-1))

print(result)
