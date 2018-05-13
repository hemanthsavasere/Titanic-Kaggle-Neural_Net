import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

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
model.add(Dense(10000, input_dim=5, init='uniform', activation='relu'))
model.add(Dropout(0.75)) 
model.add(Dense(250, init='uniform', activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer= optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.0),
              metrics=['accuracy'])
model.fit(features,target, nb_epoch=100, batch_size=10,  verbose=1)

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

prediction = [int(x) for x in prediction]
print(prediction)


rounded = [int(x > 0.425 ) for x in prediction]
print(rounded)
print (len(rounded))

my_solution = pd.DataFrame({'PassengerId': test["PassengerId"], 'Survived':prediction  })
print(my_solution)
my_solution.to_csv("submission2.csv", index = False)

