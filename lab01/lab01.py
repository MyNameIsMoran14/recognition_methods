"""ЛР № 1. Признаковые описания и метрики.

Вариант 4: p = 4 (метрика Минковского в шаге 4), lambda = -2 (K-расстояние
по Колмогорову). Дополнительное задание — расстояние Канберры между
классами Iris и его отличие от d2.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

RANDOM_STATE = 42
VARIANT = 4
MINKOWSKI_P = 4
KOLMOGOROV_LAMBDA = -2

np.set_printoptions(precision=4, suppress=True)


def d_minkowski(x, y, p=2.0):
    """Метрика Минковского d_p.

    При p=2 - Евклид; p=1 - Манхэттен; p -> inf - доминирование.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.power(np.sum(np.abs(x - y) ** p, axis=-1), 1.0 / p)


def d_chebyshev(x, y):
    """Метрика доминирования d_inf."""
    return np.max(np.abs(np.asarray(x) - np.asarray(y)), axis=-1)


def d_canberra(x, y):
    """Метрика Канберра d_k."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    num = np.abs(x - y)
    den = np.abs(x) + np.abs(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / den, 0.0)
    return np.sum(ratio, axis=-1)


def cosine_dist(x, y):
    """Косинусное расстояние."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    num = np.sum(x * y, axis=-1)
    den = np.sqrt(np.sum(x * x, axis=-1)) * np.sqrt(np.sum(y * y, axis=-1))
    cos = np.clip(num / den, -1.0, 1.0)
    return np.arccos(cos)


def class_stats(X_class):
    """Вычислить m и S класса.

    X_class: массив (m, n), где строки - образы класса.
    """
    X_class = np.asarray(X_class, dtype=float)
    m = X_class.mean(axis=0)
    Xc = X_class - m
    S = Xc.T @ Xc / X_class.shape[0]
    return m, S


def d_mahalanobis(x, mean, S_inv):
    """Метрика Махаланобиса до класса."""
    v = np.asarray(x, dtype=float) - np.asarray(mean, dtype=float)
    return v @ S_inv @ v


def nearest_mean_classify(X_train, y_train, X_test, metric="euclid"):
    """Классификатор ближайшего центра. Возвращает предсказанные метки."""
    classes = np.unique(y_train)
    means = np.stack([X_train[y_train == c].mean(axis=0) for c in classes])
    if metric == "mahalanobis":
        Ss = [class_stats(X_train[y_train == c])[1] for c in classes]
        S_bar = np.mean(Ss, axis=0)
        S_inv = np.linalg.pinv(S_bar)
        dists = np.stack(
            [[d_mahalanobis(x, mu, S_inv) for mu in means] for x in X_test]
        )
    else:
        dists = np.linalg.norm(
            X_test[:, None, :] - means[None, :, :], axis=2
        )
    return classes[np.argmin(dists, axis=1)]


if __name__ == "__main__":
    iris = load_iris()
    wine = load_wine()
    print("Iris:", iris.data.shape, "классов:", len(iris.target_names))
    print("Wine:", wine.data.shape, "классов:", len(wine.target_names))

    X_tr, X_te, y_tr, y_te = train_test_split(
        wine.data, wine.target, test_size=0.25, stratify=wine.target,
        random_state=RANDOM_STATE,
    )
    for metric in ("euclid", "mahalanobis"):
        y_hat = nearest_mean_classify(X_tr, y_tr, X_te, metric=metric)
        print(f"metric={metric}: accuracy={accuracy_score(y_te, y_hat):.3f}")
        print(confusion_matrix(y_te, y_hat))
