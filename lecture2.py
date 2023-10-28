import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import seaborn as sns

#%%
excel_file = "dataset_book_prediction.xlsx"
excel_data = pd.read_excel(excel_file)

csv_file = "dataset_book_prediction.csv"
excel_data.to_csv(csv_file, index = False)

df = pd.read_csv(csv_file)

#plt.scatter(df.Cinsiyet, df.Egitim_durumu)
#plt.show()

string_to_integer_mapping = {}

for column in df.columns:
    if df[column].dtype == 'object':
        unique_values = df[column].unique()
        for value in unique_values:
            if value not in string_to_integer_mapping:
                string_to_integer_mapping[value] = np.random.randint(1, 100000)
        df[column] = df[column].map(string_to_integer_mapping)

new_csv_file_path = 'encoding.csv'
df.to_csv(new_csv_file_path, index = False)

#%% Splitting Data
train_df, test_df = train_test_split(df, test_size = 0.3, random_state = 42)

x_train = train_df.drop(["en_sevdigi_kitap"], axis = 1)
y_train = train_df["en_sevdigi_kitap"]

x_test = test_df.drop(["en_sevdigi_kitap"], axis = 1)
y_test = test_df["en_sevdigi_kitap"]

y_train = y_train.astype('int')
y_test = y_test.astype('int')

print("x_train", len(x_train))
print("y_train", len(y_train))
print("x_test", len(x_test))
print("y_test", len(y_test))

#%% Linear Regression
model_Linear = LinearRegression()

model_Linear.fit(x_train, y_train)

predictions_Linear = model_Linear.predict(x_test)

print("Linear Regression Score:", model_Linear.score(x_test, y_test))

plt.scatter(y_test, predictions_Linear, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Real Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression - Real vs Prediction")
plt.show()

#%% KNN
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model_KNN = KNeighborsClassifier(n_neighbors=3)
model_KNN.fit(x_train, y_train)

predictions_KNN = model_KNN.predict(x_test)

model_ConfusionMatrix = confusion_matrix(y_test, predictions_KNN)
classLabels = ["Harry Potter", "Sapiens", "Çiçek Senfonisi", "Alamut Kalesi", "Ben Robot", "Diğer"] 
sns.heatmap(model_ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=classLabels, yticklabels=classLabels)

plt.title("Confusion Matrix")
plt.xlabel("Prediction")
plt.ylabel("Real")
plt.show()

accuracy_KNN = accuracy_score(y_test, predictions_KNN)
print("KNN Accuracy Score:", accuracy_KNN)
#f1_KNN = f1_score(y_test, predictions_KNN)
#print("F1:", f1_KNN), predictions_KNN)
#print("F1:", f1_KNN)#%%
