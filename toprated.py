import pandas
import numpy as np
from sys import argv

file_path = argv[1]
film_name_path = argv[2]
_old = argv[3]

old = False
if _old == "1":
    old = True

name = 'title'
if old:
   name = 'name'



ratings = pandas.read_csv(file_path)
movies = None
if old:
    movies = pandas.read_csv(film_name_path,sep='|',usecols=range(5))
else:
    movies = pandas.read_csv(film_name_path)


df = pandas.merge(ratings,movies)
group = df.groupby([name,'movieId'])['rating'].sum().reset_index()
sort = group.sort_values('rating',ascending=False)
print(sort[:10])

result=[]
for i,v in sort[:5].iterrows():
    result.append((v['movieId'],-1))

print(result)
