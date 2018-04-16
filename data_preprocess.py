import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler
train = pd.read_csv("Data.csv")
train["Country"] = LabelEncoder().fit_transform(train["Country"])
train["Salary"] = train["Salary"].fillna(train["Salary"].mean())
train["Age"] = train["Age"].fillna(train["Age"].mean())
train = pd.get_dummies(train, columns=["Country"])
features = train[["Country_0","Country_1","Country_2","Age","Salary"]].values
train["Purchased"] = LabelEncoder().fit_transform(train["Purchased"])
target = train[["Purchased"]].values
features = StandardScaler().fit_transform(features)

from sklearn.model_selection import train_test_split
train, test = train_test_split(features, train_size = 0.8)
