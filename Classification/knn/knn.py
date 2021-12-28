import numpy as np
import tqdm
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from typing import Union



def _distancia_euclideana(x_1: np.array, x_2: np.array):
    ''' DISTANCIA EUCLIDEANA '''
    return np.sum((x_1 - x_2) ** 2)


def _distancia_manhattan(x_1: np.array, x_2: np.array):
    ''' DISTANCIA MANHATTAN '''
    return np.sum(np.abs(x_1 - x_2))


def _calcula_distancias(X_train: np.array, x_2: np.array, nombre_distancia: str) -> np.array:
    ''' CALCULA LA DISTANCIA DE UN VECTOR Y UNA MATRIZ '''
    distancia_elegida = _distancia_euclideana if nombre_distancia == 'euclidean' else _distancia_manhattan
    distancias = [distancia_elegida(X_train[indice, :], x_2) for indice in range(X_train.shape[0])]
    return np.array(distancias)


def _devuelve_cercanos(distancias: np.array, K: int) -> np.array:
    ''' DEVUELVE LOS K INDICES MÁS CERCANOS '''
    return np.argsort(distancias)[:K]


def _predice_clase_cercana(y_train: np.array, indices: np.array) -> Union[str, int]:
    ''' PREDICE LA CLASE MÁS CERCANA '''
    values, counts = np.unique(y_train[indices], return_counts=True)
    ind = np.argmax(counts)
    return values[ind]


def _knn_una_observacion(X_train: np.array, y_train: np.array, x_2: np.array, nombre_distancia: str, K: int) -> Union[str, int]:
    ''' KNN PARA UNA OBSERVACION '''
    distancias = _calcula_distancias(X_train, x_2, nombre_distancia)
    cercanos = _devuelve_cercanos(distancias, K)
    clase_predicha = _predice_clase_cercana(y_train, cercanos)
    return clase_predicha


def knn_clasificacion(X_train, y_train, X_test, nombre_distancia, K) -> np.array:
    ''' KNN PARA UNA MATRIZ '''
    clases = [_knn_una_observacion(X_train, y_train, X_test[indice,:], nombre_distancia, K) for indice in tqdm.tqdm(range(X_test.shape[0]))]
    return np.array(clases)


if __name__ == '__main__':

    ''' GENERANDO DATOS '''
    X, y = make_classification(n_samples=1000,
                               n_features=2,
                               n_informative=2,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=4,
                               n_clusters_per_class=1,
                               random_state=2022)

    ''' DIVIDIENDO LOS DATOS EN ENTRENAMIENTO Y TEST '''
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=123456,
                                                        test_size=0.2)


    ''' LISTA DE COLORES DE LAS CLASES '''
    LISTA_COLORES = ['b', 'g', 'r', 'c']


    ''' GRAFICANDO CONJUNTO ENTRENAMIENTO '''
    for clase in range(4):
        plt.scatter(X_train[y_train == clase, 0], X_train[y_train == clase, 1], c=LISTA_COLORES[clase], s=100)
        plt.title("Conjunto de entrenamiento")


    ''' GRAFICANDO CONJUNTO PRUEBA '''
    for clase in range(4):
        plt.scatter(X_test[y_test == clase, 0], X_test[y_test == clase, 1], c=LISTA_COLORES[clase], s=100)
        plt.title("Conjunto de prueba")




    ''' ELIGIENDO UNA OBSERVACION '''
    x_2 = X_test[100, :]
    
    for i in range(X_train.shape[0]):
        target = y_train[i]
        color = 'b' if target == 0 else 'g' if target == 1 else 'r' if target == 2 else 'c'
        plt.scatter(x_2[0], x_2[1])
        plt.plot([x_2[0], X_train[i, 0]], [x_2[1], X_train[i, 1]], color + 'o-', linewidth=0.5, markersize=0.2)


    distancias = _calcula_distancias(X_train, x_2, 'euclidean')


    cercanos = _devuelve_cercanos(distancias, 15)

    for i in cercanos:
        target = y_train[i]
        color = 'b' if target == 0 else 'g' if target == 1 else 'r' if target == 2 else 'c'
        plt.scatter(x_2[0], x_2[1])
        plt.plot([x_2[0], X_train[i, 0]], [x_2[1], X_train[i, 1]], color + 'o-', linewidth=2, markersize=2)

    plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c='b', s=0.5)
    plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c='g', s=0.5)
    plt.scatter(X_train[y_train == 2, 0], X_train[y_train == 2, 1], c='r', s=0.5)
    plt.scatter(X_train[y_train == 3, 0], X_train[y_train == 3, 1], c='c', s=0.5)



    clases_predichas = knn_clasificacion(X_train, y_train, X_test, 'euclidean', 15)


    plt.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c='b')
    plt.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c='g')
    plt.scatter(X_test[y_test == 2, 0], X_test[y_test == 2, 1], c='r')
    plt.scatter(X_test[y_test == 3, 0], X_test[y_test == 3, 1], c='c')


    plt.scatter(X_test[clases_predichas == 0, 0], X_test[clases_predichas == 0, 1], c='b')
    plt.scatter(X_test[clases_predichas == 1, 0], X_test[clases_predichas == 1, 1], c='g')
    plt.scatter(X_test[clases_predichas == 2, 0], X_test[clases_predichas == 2, 1], c='r')
    plt.scatter(X_test[clases_predichas == 3, 0], X_test[clases_predichas == 3, 1], c='c')













