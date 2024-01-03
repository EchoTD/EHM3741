"""
    EHM3741 Gömülü Sistemlerde Makine Öğrenmesi | Proje
    Alaaddin Can Gürsoy | 21014506
    
"""
#%% Import required libraries and setup the data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

plotFolder = "/home/acg/EHM3741/HomeworkProject/plotFolder"
csvFile = "/home/acg/EHM3741/HomeworkProject/Iris.csv"
dataFrame = pd.read_csv(csvFile)

label_encoder = LabelEncoder()
dataFrame['Species'] = label_encoder.fit_transform(dataFrame['Species'])

#%% Splitting the data to testing and training data sets
train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.3, random_state=21)

trainFeatures = train_dataFrame.drop(["Species"], axis = 1)
trainTarget = train_dataFrame["Species"]
testFeatures = test_dataFrame.drop(["Species"], axis = 1)
testTarget = test_dataFrame["Species"]

#%% Linear Regression method
from sklearn.linear_model import LinearRegression
model_Linear = LinearRegression()

model_Linear.fit(trainFeatures, trainTarget)
prediction_linear = model_Linear.predict(testFeatures)

cv_score_Linear = cross_val_score(model_Linear, trainFeatures, trainTarget, cv=6)
print("Linear Regression CV Scores:", cv_score_Linear)
print("Average CV Score for Linear Regression:", np.mean(cv_score_Linear))
print("Linear Regression Score:", model_Linear.score(trainFeatures, trainTarget))

predicted_labels = label_encoder.inverse_transform(prediction_linear.astype(int))
actual_labels = label_encoder.inverse_transform(testTarget)

plt.scatter(actual_labels, predicted_labels, color='green')
plt.plot([min(actual_labels), max(actual_labels)], [min(actual_labels), max(actual_labels)], color='red')
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Real vs Prediction")
fileName = "linearPlot.png"
fullPath = os.path.join(plotFolder, fileName)
plt.savefig(fullPath)

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
trainFeatures_scaled = scaler.fit_transform(trainFeatures)
testFeatures_scaled = scaler.transform(testFeatures)

# 21014506 -> A = 6 -> n_neighbors = 6
model_KNN = KNeighborsClassifier(n_neighbors=6)
model_KNN.fit(trainFeatures_scaled, trainTarget)
prediction_KNN = model_KNN.predict(testFeatures_scaled)

cv_score_KNN = cross_val_score(model_KNN, trainFeatures_scaled, trainTarget, cv=6)
print("KNN (n=6) CV Scores:", cv_score_KNN)
print("Average CV Score for KNN (n=6)", np.mean(cv_score_KNN))

accuracy_KNN = accuracy_score(testTarget, prediction_KNN)
print("KNN (n=6) Accuracy Score:", accuracy_KNN)

species_labels = label_encoder.inverse_transform(np.unique(testTarget))
KNN_ConfusionMatrix = confusion_matrix(testTarget, prediction_KNN, labels=np.unique(testTarget))
sns.heatmap(KNN_ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=species_labels, yticklabels=species_labels)
plt.title("KNN (n=6) - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
fileName = "knn(6).png"
fullPath = os.path.join(plotFolder, fileName)
plt.savefig(fullPath)

# 21014506 -> B = 0, can't use 0, use default "2" -> n_neighbors = 2
model_KNN = KNeighborsClassifier(n_neighbors=2)
model_KNN.fit(trainFeatures_scaled, trainTarget)
prediction_KNN = model_KNN.predict(testFeatures_scaled)

cv_score_KNN = cross_val_score(model_KNN, trainFeatures_scaled, trainTarget, cv=6)
print("KNN (n=2) CV Scores:", cv_score_KNN)
print("Average CV Score for KNN (n=2)", np.mean(cv_score_KNN))

accuracy_KNN = accuracy_score(testTarget, prediction_KNN)
print("KNN (n=2) Accuracy Score:", accuracy_KNN)

species_labels = label_encoder.inverse_transform(np.unique(testTarget))
KNN_ConfusionMatrix = confusion_matrix(testTarget, prediction_KNN, labels=np.unique(testTarget))
sns.heatmap(KNN_ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=species_labels, yticklabels=species_labels)
plt.title("KNN (n=2) - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
fileName = "knn(2).png"
fullPath = os.path.join(plotFolder, fileName)
plt.savefig(fullPath)


