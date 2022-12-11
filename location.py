import matplotlib.pyplot as plt
import csv
from math import sqrt
from tqdm.auto import tqdm
import pandas as pd
import os

stat = pd.read_csv("C:/Users/wayne/Desktop/town/towninfo.csv")
townlist = pd.read_csv("C:/Users/wayne/Desktop/town/townlist.csv")

name = stat["TOWNNAME1"]
item = stat["name1"]

townlist = townlist["townlist"]
townlist_arr = []
for i in range(len(townlist)):
  townlist_arr.append(townlist[i])

statlist = []

x = '七股區'
arr = []
for i in range(len(item)):
  print(i)
  if(name[i]!=x):
    x = name[i]
    statlist.append(arr)
    arr = []
    arr.append(item[i])
  else:
    arr.append(item[i])
statlist.append(arr)

print(townlist_arr)