import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense

def read_dataset(test, train):
    train = pd.read_csv(train)
    test = pd.read_csv(test)
    return test, train

def encode(columns):
    columns = LabelEncoder().fit_transform(columns)
    return columns

def family_size(n):
    if n < 1:
        return 0
    else:
        return 1

# laoding testing and training dataset
test, train = read_dataset("test.csv", "train.csv")

# filling the missing data with mean of age
train["Age"] = train["Age"].fillna(train["Age"].mean())
#encoding the data
train['Sex'] = encode(train['Sex'])
# filling the missing data
train["Embarked"] = train["Embarked"].fillna("S")
#encoding the data 
train['Embarked'] = encode(train['Embarked'])

#creating a new feature family size that is sum of Parents and Siblings and himself
train["Family_size"] = train["Parch"] + train["SibSp"]
train['Family_size']=list(map(family_size, train['Family_size'] ))
features = train[["Pclass","Sex","Age","Fare","Family_size"]].values

# using standard scaler to scale the data 
features = StandardScaler().fit_transform(features)
target = train[["Survived"]].values
print (features)


model = Sequential()
model.add(Dense(1000, input_dim=5, init='uniform', activation='relu')) 
model.add(Dense(500, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(features,target, nb_epoch=150, batch_size=5,  verbose=2)

test["Embarked"] = encode(test["Embarked"])
test["Sex"] = encode(test["Sex"])
test["Family_size"] = test["Parch"] + test["SibSp"]
test['Family_size']=list(map(family_size, test['Family_size']))
test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())


test_features = test[["Pclass","Sex","Age","Fare","Family_size"]].values  
 #using standard scaler for test features
test_features = StandardScaler().fit_transform(test_features)

prediction = model.predict_classes(test_features)

print(prediction)

my_solution = pd.DataFrame({'PassengerId': test["PassengerId"], 'Survived': prediction[:,0] })
print(my_solution)
my_solution.to_csv("submission.csv", index = False)

