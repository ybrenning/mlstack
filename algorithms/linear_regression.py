import numpy as np


class LinearRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(0, self.n_iters):
            y_pred = np.dot(X, self.w) + self.b
            dw = 1 / n_samples * np.dot((y_pred - y), X)
            db = 1 / n_samples * np.sum(y_pred - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(lr=0.01)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)

    def mse(y_t, y_p): return np.mean((y_t - y_p) ** 2)

    print(mse(y_test, predicted))

    plt.scatter(X[:, 0], y, color="blue", s=10)
    plt.plot(X, regressor.predict(X), color="red", label="Prediction 1")

    # regressor2 = LinearRegression(lr=0.001)
    # regressor2.fit(X_train, y_train)

    # print(mse(y_test, regressor2.predict(X_test)))
    # plt.plot(X, regressor2.predict(X), color="magenta", label="Prediction 2")
    plt.show()
