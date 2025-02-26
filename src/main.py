import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

file = "C:\\Users\\moham\\Desktop\\DSProject\\Turnover-Prediction\\data\\hr_data.csv"

df = pd.read_csv(file)
data = df.copy()

print(data.head())
print(data.info())
print(data.isnull().sum())
print(data.describe())

print(data["Attrition"].value_counts())
label_encoder = LabelEncoder()
data["Attrition"] = label_encoder.fit_transform(data["Attrition"])

cols_to_drop = ["EmployeeNumber", "Over18", "StandardHours"]
data.drop(cols_to_drop, axis = 1, inplace = True)

cat_columns = [col for col in data.columns if data[col].dtypes == 'object']
print(cat_columns)

y = data.Attrition
X = data.drop(["Attrition"], axis = 1)

X = pd.get_dummies(X, columns = cat_columns, drop_first = True)
print(data.head())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 1)

model = RandomForestClassifier(n_estimators = 100, random_state = 1)
model.fit(X_train, y_train)

with open("C:\\Users\\moham\\Desktop\\DSProject\\Turnover-Prediction\\models\\random_forest.pkl", "wb") as f:
    pickle.dump(model, f)

y_pred = model.predict(X_val)

print("Prédiction : ")
print(y_pred)

cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Resté", "Parti"], yticklabels=["Resté", "Parti"])
plt.xlabel("Prédictions")
plt.ylabel("Vraie valeur")
plt.title("Matrice de Confusion")
plt.show()









