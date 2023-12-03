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


csvFile = "/home/acg/.repos/EHM3741/Homework Project/Iris.csv"
dataFrame = pd.read_csv(csvFile)

label_encoder = LabelEncoder()
dataFrame['Species'] = label_encoder.fit_transform(dataFrame['Species'])

#%% Splitting the data to testing and training data sets
train_dataFrame, test_dataFrame = train_test_split(dataFrame, test_size=0.2, random_state=21)

trainFeatures = train_dataFrame.drop(["Species"], axis = 1)
trainTarget = train_dataFrame["Species"]
testFeatures = test_dataFrame.drop(["Species"], axis = 1)
testTarget = test_dataFrame["Species"]

#%% Linear Regression method
from sklearn.linear_model import LinearRegression
model_Linear = LinearRegression()

model_Linear.fit(trainFeatures, trainTarget)
prediction_linear = model_Linear.predict(testFeatures)

print("Linear Regression Score:", model_Linear.score(trainFeatures, trainTarget))

predicted_labels = label_encoder.inverse_transform(prediction_linear.astype(int))
actual_labels = label_encoder.inverse_transform(testTarget)

plt.scatter(actual_labels, predicted_labels, color='green')
plt.plot([min(actual_labels), max(actual_labels)], [min(actual_labels), max(actual_labels)], color='red')
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Real vs Prediction")
plt.show()

#%% KNN
from sklearn.neighbors import KNeighborsClassifier
scaler = StandardScaler()
trainFeatures_scaled = scaler.fit_transform(trainFeatures)
testFeatures_scaled = scaler.transform(testFeatures)


k_range = range(1, 31)
k_scores = []

# Perform 10-fold cross-validation for each value of k
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, trainFeatures_scaled, trainTarget, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Find the value of k with the highest score
best_k = k_range[k_scores.index(max(k_scores))]
print(f"Best k: {best_k} with score: {max(k_scores)}")

model_KNN = KNeighborsClassifier(n_neighbors=best_k)
model_KNN.fit(trainFeatures_scaled, trainTarget)
prediction_KNN = model_KNN.predict(testFeatures_scaled)

accuracy_KNN = accuracy_score(testTarget, prediction_KNN)
print("KNN Accuracy Score:", accuracy_KNN)

species_labels = label_encoder.inverse_transform(np.unique(testTarget))
KNN_ConfusionMatrix = confusion_matrix(testTarget, prediction_KNN, labels=np.unique(testTarget))
sns.heatmap(KNN_ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=species_labels, yticklabels=species_labels)
plt.title("KNN - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#%% SVM
from sklearn.svm import SVC
model_SVM = SVC(kernel="linear", C=1.0, random_state=42)
model_SVM.fit(trainFeatures_scaled, trainTarget)
prediction_SVM = model_SVM.predict(testFeatures_scaled)

accuracy_SVM = accuracy_score(testTarget, prediction_SVM)
print("SVM Accuracy Score:", accuracy_SVM)

#species_labels = label_encoder.inverse_transform(np.unique(testTarget))
SVM_ConfusionMatrix = confusion_matrix(testTarget, prediction_SVM, labels=np.unique(testTarget))
sns.heatmap(SVM_ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=species_labels, yticklabels=species_labels)
plt.title("SVM - Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

#%% Decision Tree Method
from sklearn.tree import DecisionTreeClassifier

model_DT = DecisionTreeClassifier(random_state=42)
model_DT.fit(trainFeatures, trainTarget)
prediction_DT = model_DT.predict(testFeatures)

accuracy_DT = accuracy_score(testTarget, prediction_DT)
print("Decision Tree Accuracy Score:", accuracy_DT)

# Confusion Matrix
DT_ConfusionMatrix = confusion_matrix(testTarget, prediction_DT)
sns.heatmap(DT_ConfusionMatrix, annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predictions")
plt.ylabel("Real")
plt.show()

#%% Naive Bayes Method
from sklearn.naive_bayes import GaussianNB

model_NB = GaussianNB()
model_NB.fit(trainFeatures, trainTarget)
prediction_NB = model_NB.predict(testFeatures)

accuracy_NB = accuracy_score(testTarget, prediction_NB)
print("Naive Bayes Accuracy Score:", accuracy_NB)

NB_ConfusionMatrix = confusion_matrix(testTarget, prediction_NB)
sns.heatmap(NB_ConfusionMatrix, annot=True, fmt='d', cmap='Blues')
plt.title("Naive Bayes - Confusion Matrix")
plt.xlabel("Predictions")
plt.ylabel("Real")
plt.show()