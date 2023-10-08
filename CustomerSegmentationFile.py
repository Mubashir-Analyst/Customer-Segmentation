# importing required libraries/dependencies

import numpy as np
import pandas as pnd
import matplotlib.pyplot as plot
import seaborn as sea
from sklearn.cluster import KMeans

# loading dataset
dataSet = pnd.read_csv('C:/Users/City College/Downloads/MallDataSet.csv.csv')

# viewing first five rows
print(dataSet.head())

# checking total number of rows and columns in dataset
print(dataSet.shape)

# getting additional information of dataset
print(dataSet.info())
print(dataSet.describe())

# looking for any null values in dataset
print(dataSet.isnull().sum())

# plotting boxPlot to visualize correlation between annual Income and Spending
plot.figure(figsize=(15,6))
plot.subplot(1,2,1)
sea.boxplot(y=dataSet["customerSpending (1-100)"], color="red")
plot.subplot(1,2,2)
sea.boxplot(y=dataSet["customerIncome"])
plot.show()

# plotting distribution of male and female in dataset
plot.figure(figsize=(8,8))
genders = dataSet.customerGender.value_counts()
sea.set_style("darkgrid")
plot.figure(figsize=(10,4))
sea.barplot(x=genders.index, y=genders.values)
plot.show()

# finding correlation between variables of dataset
plot.figure(figsize=(8, 8))
sea.heatmap(dataSet.corr(), annot=True, cmap='RdBu')
plot.title('Correlation Map', fontsize=14)
plot.yticks(rotation=0)
plot.show()


# selecting customer income and spending columns
X = dataSet.iloc[:, [3, 4]].values
print(X)

# finding right number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

# drawing an elbow graph
plot.figure(figsize=(8, 8))
sea.set()
plot.plot(range(1, 11), wcss)
plot.title('Elbow Graph')
plot.xlabel('Nos of Clusters')
plot.ylabel('WCSS')
plot.show()

# training the kmeans model
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# provides a label for every data point depending on it's cluster
Y = kmeans.fit_predict(X)
print(Y)

# plotting the clusters
plot.figure(figsize=(8, 8))
plot.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Group 1')
plot.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Group 2')
plot.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Group 3')
plot.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='violet', label='Group 4')
plot.scatter(X[Y == 4, 0], X[Y == 4, 1], s=50, c='blue', label='Group 5')

plot.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='cyan', label='Centroids')

plot.title('Customer Groups')
plot.xlabel('Customer Income')
plot.ylabel('Customer Spending')
plot.show()

# 3D view of clusters

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)
Y = kmeans.fit_predict(X)

clusters = kmeans.fit_predict(X)
dataSet["label"] = clusters

fig = plot.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dataSet.customerAge[dataSet.label == 0], dataSet["customerIncome"][dataSet.label == 0], dataSet["customerSpending (1-100)"][dataSet.label == 0],
           c='blue', s=60)
ax.scatter(dataSet.customerAge[dataSet.label == 1], dataSet["customerIncome"][dataSet.label == 1], dataSet["customerSpending (1-100)"][dataSet.label == 1],
           c='red', s=60)
ax.scatter(dataSet.customerAge[dataSet.label == 2], dataSet["customerIncome"][dataSet.label == 2], dataSet["customerSpending (1-100)"][dataSet.label == 2],
           c='green', s=60)
ax.scatter(dataSet.customerAge[dataSet.label == 3], dataSet["customerIncome"][dataSet.label == 3], dataSet["customerSpending (1-100)"][dataSet.label == 3],
           c='orange', s=60)
ax.scatter(dataSet.customerAge[dataSet.label == 4], dataSet["customerIncome"][dataSet.label == 4], dataSet["customerSpending (1-100)"][dataSet.label == 4],
           c='purple', s=60)
ax.view_init(30, 185)
plot.xlabel("Customer Age")
plot.ylabel("Customer Income k$")
ax.set_zlabel('Customer Spending (1-100)')
plot.show()
