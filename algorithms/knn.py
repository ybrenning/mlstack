from collections import Counter

import numpy as np

# See also: https://en.wikipedia.org/wiki/Euclidean_distance
ed = lambda x, y: np.sqrt(np.sum((x - y)**2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # Compute the euclidean distances
        distances = [ed(x, x_train) for x_train in self.X_train]

        # Get the k-nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Get the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    classifier = KNN(k=5)
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
    plt.show()
