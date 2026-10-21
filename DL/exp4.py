import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("play_tennis.csv")

if 'day' in data.columns:
    data = data.drop(columns=['day'])

encoders = {}

for col in data.columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    encoders[col] = le

X = data[['outlook', 'temp', 'humidity', 'wind']]
y = data['play']

model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=42
)

model.fit(X, y)

new_sample = pd.DataFrame(
    [[2, 1, 0, 1]],
    columns=['outlook', 'temp', 'humidity', 'wind']
)

prediction = model.predict(new_sample)

predicted_class = encoders['play'].inverse_transform(prediction)

print("Predicted Class:", predicted_class[0])
print("Decision Tree Accuracy:", model.score(X, y))