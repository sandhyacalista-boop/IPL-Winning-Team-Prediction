import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("ipl.csv")

# Select required columns
df = df[['team1','team2','toss_winner','toss_decision','city','winner']]

# Remove missing values
df = df.dropna()

# Label Encoding
le_team1 = LabelEncoder()
le_team2 = LabelEncoder()
le_toss_winner = LabelEncoder()
le_toss_decision = LabelEncoder()
le_city = LabelEncoder()
le_winner = LabelEncoder()

df['team1'] = le_team1.fit_transform(df['team1'])
df['team2'] = le_team2.fit_transform(df['team2'])
df['toss_winner'] = le_toss_winner.fit_transform(df['toss_winner'])
df['toss_decision'] = le_toss_decision.fit_transform(df['toss_decision'])
df['city'] = le_city.fit_transform(df['city'])
df['winner'] = le_winner.fit_transform(df['winner'])

# Features and Target
X = df[['team1','team2','toss_winner','toss_decision','city']]
y = df['winner']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy*100,2), "%")

# -------------------------
# Prediction Section
# -------------------------

print("\nAvailable Teams:")
for i, team in enumerate(le_team1.classes_):
    print(i, "-", team)

print("\nAvailable Cities:")
for i, city in enumerate(le_city.classes_):
    print(i, "-", city)

team1 = int(input("\nEnter Team1 Number: "))
team2 = int(input("Enter Team2 Number: "))
toss_winner = int(input("Enter Toss Winner Number: "))

print("\nToss Decision")
print("0 - bat")
print("1 - field")

toss_decision = int(input("Enter Toss Decision: "))

city = int(input("Enter City Number: "))

sample = [[
    team1,
    team2,
    toss_winner,
    toss_decision,
    city
]]

prediction = model.predict(sample)

winner = le_winner.inverse_transform(prediction)

print("\n🏆 Predicted Winner:", winner[0])