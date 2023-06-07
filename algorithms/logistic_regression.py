import warnings

import numpy as np

warnings.filterwarnings("ignore")

sig = lambda x: 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0

        # Gradient descent
        for _ in range(0, self.n_iters):
            linear_model = np.dot(X, self.w) + self.b
            y_pred = sig(linear_model)

            dw = 1 / n_samples * np.dot((y_pred - y), X)
            db = 1 / n_samples * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        y_pred = sig(linear_model)

        return [1 if y > 0.5 else 0 for y in y_pred]


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)

    print(acc)
