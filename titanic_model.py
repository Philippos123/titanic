import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Ladda in Titanic-datasetet
data = pd.read_csv('train.csv')

# Dataförberedelse
# Fyll i saknade värden
data['Age'] = data['Age'].fillna(data['Age'].median())  # Ålder
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])  # Embarked

# Omvandla kön till numeriska värden
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Välj relevanta funktioner och mål
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']  # Funktioner
X = data[features]
y = data['Survived']  # Målvariabel

# Dela upp data i tränings- och testuppsättningar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Träna modellen
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Gör förutsägelser
y_pred = model.predict(X_test)

# Utvärdera modellen
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Spara modellen för framtida användning
joblib.dump(model, 'titanic_model.pkl')

# Ladda modellen igen (om du vill)
# loaded_model = joblib.load('titanic_model.pkl')
