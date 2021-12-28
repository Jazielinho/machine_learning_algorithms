from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from typing import Union


X, y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=4,
                           n_clusters_per_class=1, random_state=123456)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123456, test_size=0.2)


knn = KNeighborsClassifier(n_neighbors=15, algorithm='brute')


knn.fit(X_train, y_train)


clases_predichas = knn.predict(X_test)