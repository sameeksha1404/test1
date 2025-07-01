from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
X,y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=20)
model.fit(X,y)
print("model trained")