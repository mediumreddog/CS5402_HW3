from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn import datasets

# iris = datasets.load_iris()
# print(iris.data, iris.target)

train_features = [
    "location",
    "top25",
    "network"
]

# Read in CSV files
data_training = pd.read_csv("games_training.csv")
data_test = pd.read_csv("games_test.csv")

# Replace categorical features with numbers
data_training["network"]=np.where(data_training["network"]=="NBC",1,
    np.where(data_training["network"]=="ESPN",2,
        np.where(data_training["network"]=="FOX",3,4)))
data_test["network"]=np.where(data_test["network"]=="NBC",1,
    np.where(data_test["network"]=="ESPN",2,
        np.where(data_test["network"]=="FOX",3,4)))

data_training["location"]=np.where(data_training["location"]=="Away",0,1)
data_test["location"]=np.where(data_test["location"]=="Away",0,1)

data_training["top25"]=np.where(data_training["top25"]=="Out",0,1)
data_test["top25"]=np.where(data_test["top25"]=="Out",0,1)

data_training["decision"]=np.where(data_training["decision"]=="Lose",0,1)
data_test["decision"]=np.where(data_test["decision"]=="Lose",0,1)

# Set X and Y
X = data_training[train_features].values
y = data_training["decision"].values

# Create naive bayes
gnb = GaussianNB()
gnb.fit(X, y)

# Predict values with classifier
y_pred = gnb.predict(data_test[train_features])
print(y_pred)
# Print out results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          data_test.shape[0],
          (data_test["decision"] != y_pred).sum(),
          100*(1-(data_test["decision"] != y_pred).sum()/data_test.shape[0])
))

