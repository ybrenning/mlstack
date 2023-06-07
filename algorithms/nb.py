import numpy as np


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c = X[c == y]
            self._mean[c, :] = X_c.mean(axis=0)
            self._var[c, :] = X_c.var(axis=0)
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []

        for i in range(0, len(self._classes)):
            prior = np.log(self._priors[i])
            cls_cond = np.sum(np.log(self._pdf(i, x)))
            posteriors.append(prior + cls_cond)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, cls_i, x):
        mean = self._mean[cls_i]
        var = self._var[cls_i]
        return np.exp(-((x - mean) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=1234
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)

    y_preds = nb.predict(X_test)

    acc = np.sum(y_preds == y_test) / len(y_test)
    print(acc)
