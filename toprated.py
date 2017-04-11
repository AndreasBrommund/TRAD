import pandas
from sys import argv

file_path = argv[1]
film_name_path = argv[2]
old = argv[3]
sep = argv[4]

if old == "1":
    old = True
else:
    old = False

#source: http://www.gregreda.com/2013/10/26/using-pandas-on-the-movielens-dataset/
movies = None

if old:
    movies = pandas.read_csv(film_name_path, sep='|', usecols=range(5),encoding='latin-1')
else:
    movies = pandas.read_csv(film_name_path,sep=sep)


ratings = pandas.read_csv(file_path,sep=sep)
movie_ratings = pandas.merge(movies, ratings)

most_rated = []
if old:
    most_rated = movie_ratings['name'].value_counts()[:5]
else:
    most_rated = movie_ratings['title'].value_counts()[:5]

print ("Top 5 Films")
print(most_rated)
