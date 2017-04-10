import pandas
from sys import argv


path = argv[1]
#read only the films genre
dataset = pandas.read_csv(path,sep = "|",usecols=range(5,24))
