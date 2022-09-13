import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

from LinearRegression import LinearRegression

def loss_function(y_true, y_pred):
    return np.mean((y_true - y_pred)**2) * 1/2

model = LinearRegression(lr=0.005,epochs=5000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

loss = loss_function(y_test, predictions)
print(loss)

y_pred_line = model.predict(X)
cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()