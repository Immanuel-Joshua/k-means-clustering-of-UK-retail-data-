import pandas as pd
import numpy as np
from sklearn.cluster import KMeans 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


"""Data preparation"""
#Load dataset using pandas 
df = pd.read_csv('UK retailer data.csv', encoding='ISO-8859-1')
#Remove all duplicates in dataset
df = df.drop_duplicates()
print(df)
# Fill the rows with missing values
df['CustomerID'] = df['CustomerID'].fillna(0)
df['Description'] = df['Description'].fillna('Unknown')
#Check null
print(df.isnull().sum(), df.duplicated().sum)
# Remove rows with negative unit price as they ae related to debt and will skew our analysis
debtmask = df[~df['Description'].str.contains('debt', case=False, na=False)]

# Now i will create a total column which will give Quantity*UnitPrice.
df['Total'] = df['Quantity']*df['UnitPrice']
# Convert CustomerID to float fo tidyness
df['CustomerID'] = df['CustomerID'].astype(int)

#Move columns for better readability
df = df[['InvoiceNo', 'StockCode', 'Description', 'Quantity',
        'UnitPrice', 'Total', 'CustomerID',
        'InvoiceDate', 'Country']]
#Remove rows containing 0 UnitPrice
df = df[df['UnitPrice'] > 0]
#Create new datadrame of rows with quantity 0
returns = df[df['Quantity'] < 0]
print(returns.head())
#Remove cancelled orders from main dataset
X = df[df['Quantity'] > 0]
print(X.info())
#Label encode features
label_encoder = LabelEncoder()
for column in X.columns:
    if X[column].dtype == 'object':
        X[column] = label_encoder.fit_transform(X[column])

# Select sample size
X_sample = X.sample(frac=0.001, random_state=1)

#Compute silhouette scores
K_range = range(2, 11)
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X_sample)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X_sample, labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.plot(K_range, silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal k')
plt.show()


"""Write the K-means algorithm"""
num_clusters = 6

#Create Kmeans object
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

#Train Kmeans model
kmeans.fit(X_sample)

#Predict output labels for all points on the grid
output = kmeans.predict(X_sample)

#Step size of the mesh
step_size = 1.0

# Define x_min, x_max, y_min, y_max based on the data
x_min, x_max = X_sample.iloc[:, 0].min() - 1, X_sample.iloc[:, 0].max() + 1
y_min, y_max = X_sample.iloc[:, 1].min() - 1, X_sample.iloc[:, 1].max() + 1

# Check the shape of X_sample
print("X_sample shape:", X_sample.shape)

# Meshgrid for plotting decision boundaries
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                     np.arange(y_min, y_max, step_size))

# Create a DataFrame with the same column names as X_sample
dummy_data = pd.DataFrame(np.zeros((xx.ravel().shape[0], X_sample.shape[1])), columns=X_sample.columns)

# Get labels for each point in the meshgrid
mesh_labels = kmeans.predict(dummy_data)
mesh_labels = mesh_labels.reshape(xx.shape)

# Plot decision boundaries
plt.figure()
plt.contourf(xx, yy, mesh_labels, cmap=plt.cm.Paired, alpha=0.8)

#Plot input data
plt.figure()
plt.scatter(X_sample.iloc[:, 0], X_sample.iloc[:, 1], marker='o', facecolors='none',
            edgecolors='black', s=80)

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='x', s=200, linewidths=3, color='r', zorder=10)

plt.title('K-means clustering with boundaries and cluster centers')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
