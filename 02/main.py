import pandas as pd;

from sklearn.linear_model import LogisticRegression 

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# Input
train = pd.read_csv("D:/File/HW/Files/Programming/Spyder/PNU_AI/02/train.csv")
test = pd.read_csv("D:/File/HW/Files/Programming/Spyder/PNU_AI/02/test.csv")
submission = pd.read_csv("D:/File/HW/Files/Programming/Spyder/PNU_AI/02/submission.csv")

# Preprocessing
train['hour_bef_temperature'] = train['hour_bef_temperature'].fillna(16.6)
train['hour_bef_precipitation'] = train['hour_bef_precipitation'].fillna(0.015)
train['hour_bef_windspeed'] = train['hour_bef_windspeed'].fillna(2.357)
train['hour_bef_humidity'] = train['hour_bef_humidity'].fillna(51.732)
train['hour_bef_visibility'] = train['hour_bef_visibility'].fillna(1491.108)

test['hour_bef_temperature'] = test['hour_bef_temperature'].fillna(16.6)
test['hour_bef_precipitation'] = test['hour_bef_precipitation'].fillna(0.015)
test['hour_bef_windspeed'] = test['hour_bef_windspeed'].fillna(2.357)
test['hour_bef_humidity'] = test['hour_bef_humidity'].fillna(51.732)
test['hour_bef_visibility'] = test['hour_bef_visibility'].fillna(1491.108)

# Modeling
features = ['hour', 'hour_bef_temperature', 'hour_bef_windspeed']

X_train = train[features]
y_train = train['count']
X_test = test[features]

models = []

models.append(LogisticRegression())
models.append(DecisionTreeClassifier())
models.append(DecisionTreeRegressor())
models.append(RandomForestClassifier())
models.append(RandomForestRegressor(n_estimators = 100, random_state = 0))
models.append(RandomForestRegressor(n_estimators = 100, random_state = 0, max_depth = 5))
models.append(RandomForestRegressor(n_estimators = 200, random_state = 0))
models.append(RandomForestRegressor(n_estimators = 200, random_state = 0, max_depth = 5))
models.append(RandomForestRegressor(n_estimators = 50, random_state = 0))
models.append(RandomForestRegressor(n_estimators = 50, random_state = 0, max_depth = 5))

# Predict and Output
for i in range(len(models)):
    models[i].fit(X_train, y_train)
    pred = models[i].predict(X_test)
    submission['count'] = pred
    submission.to_csv('Data' + str(i) + ".csv", index = False)