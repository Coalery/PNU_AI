import pandas as pd

from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier

# Input
train = pd.read_csv("D:/File/HW/Files/Programming/Spyder/PNU_AI/01/train.csv")
test = pd.read_csv("D:/File/HW/Files/Programming/Spyder/PNU_AI/01/test.csv")
submission = pd.read_csv("D:/File/HW/Files/Programming/Spyder/PNU_AI/01/train_submission.csv")

# PreProcessing
train["Age"] = train["Age"].fillna(28) # Age 에서 빈 값을 28로 채운다.
train["Embarked"] = train["Embarked"].fillna("S") # Embarked 에서 빈 값을 S로 채운다.
train["Sex"] = train["Sex"].map({"male" : 0, "female" : 1}) # Sex 에서 male을 0으로, female을 1로 매핑한다.

test["Age"] = test["Age"].fillna(28) # Age 에서 빈 값을 28로 채운다.
test["Embarked"] = test["Embarked"].fillna("S") # Embarked 에서 빈 값을 S로 채운다.
test["Sex"] = test["Sex"].map({"male" : 0, "female" : 1}) # Sex 에서 male을 0으로, female을 1로 매핑한다.

# Modeling
X_train = train[['Sex','Pclass', 'Age']] # 성별과 좌석등급만 가져온다.
y_train = train["Survived"] # 생존 여부를 가져온다.

X_test = test[['Sex','Pclass', 'Age']] # 테스트할 데이터의 성별과 좌석등급만 가져온다.

lr = LogisticRegression()
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train) # 로지스틱 회귀 객체에 배울 데이터를 넣어준다.
dt.fit(X_train, y_train) # 결정 트리 객체에 배울 데이터를 넣어준다.

print(lr.score(X_train, y_train))
print(dt.score(X_train, y_train))

lr_predict = lr.predict_proba(X_test)[:,1] # 위에서 넣어준 데이터에 따라 예상한다.
dt_predict = dt.predict_proba(X_test)[:,1] # 위에서 넣어준 데이터에 따라 예상한다.

# Output
submission["Survived"] = lr_predict
submission.to_csv('logistic_regression_pred.csv', index = False)

submission["Survived"] = dt_predict
submission.to_csv('decision_tree_pred.csv', index = False)
