"""ЛР № 1. Знакомство со средой: Python, NumPy, matplotlib, scikit-learn.

Вариант 4: K = 3 (petal width) для гистограммы, J = 42 для цифры из Digits.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.metrics import pairwise_distances

FEATURE_INDEX = 3  # K: номер признака Iris для гистограммы (вариант 4)
DIGIT_INDEX = 42   # J: индекс цифры из Digits (вариант 4)

# --- ШАГ 1: загрузка трёх датасетов ---
iris = load_iris()
wine = load_wine()
digits = load_digits()

print("Iris X.shape =", iris.data.shape, "классов:", len(iris.target_names))
print("Wine X.shape =", wine.data.shape, "классов:", len(wine.target_names))
print("Digits X.shape =", digits.data.shape, "классов:", len(digits.target_names))
print("Iris DESCR[:300]:\n", iris.DESCR[:300])
print("Wine DESCR[:300]:\n", wine.DESCR[:300])
print("Digits DESCR[:300]:\n", digits.DESCR[:300])

# --- ШАГ 2: изучение полей Bunch на примере Iris ---
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("Признаки:", feature_names)
print("Классы:", target_names)
print("Средние по признакам:", X_iris.mean(axis=0).round(3))

# --- ШАГ 3: гистограмма признака K по трём классам ---
plt.figure(figsize=(6, 4))
for cls in range(len(target_names)):
    values = X_iris[y_iris == cls, FEATURE_INDEX]
    plt.hist(values, bins=15, alpha=0.5, label=target_names[cls])
plt.xlabel(feature_names[FEATURE_INDEX])
plt.ylabel("Число объектов")
plt.title(f"Гистограмма признака '{feature_names[FEATURE_INDEX]}' по классам")
plt.legend()
plt.tight_layout()
plt.savefig("hist.png", dpi=120)
plt.close()

# --- ШАГ 4: диаграмма рассеяния Iris (petal length vs petal width) ---
plt.figure(figsize=(6, 5))
for cls in range(len(target_names)):
    mask = y_iris == cls
    plt.scatter(X_iris[mask, 2], X_iris[mask, 3], label=target_names[cls])
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.title("Iris: длина против ширины лепестка")
plt.legend()
plt.tight_layout()
plt.savefig("scatter.png", dpi=120)
plt.close()

# --- ШАГ 5: изображение цифры J из Digits ---
digit_vector = digits.data[DIGIT_INDEX]
digit_image = digit_vector.reshape(8, 8)
digit_label = digits.target[DIGIT_INDEX]

plt.figure(figsize=(3, 3))
plt.imshow(digit_image, cmap="gray")
plt.title(f"Цифра №{DIGIT_INDEX}, метка = {digit_label}")
plt.axis("off")
plt.tight_layout()
plt.savefig("digit.png", dpi=120)
plt.close()

# --- ШАГ 6: ручной расчёт евклидова расстояния между цветками ---
x1 = X_iris[0]
x2 = X_iris[100]
d_manual = np.sqrt(np.sum((x1 - x2) ** 2))
d_sklearn = pairwise_distances(x1.reshape(1, -1), x2.reshape(1, -1), metric="euclidean")[0, 0]
print("d_2(x0, x100) вручную =", round(d_manual, 3))
print("d_2(x0, x100) через sklearn =", round(d_sklearn, 3))

x_same_a = X_iris[0]
x_same_b = X_iris[20]
d_same_class = np.sqrt(np.sum((x_same_a - x_same_b) ** 2))
print("d_2(x0, x20), один класс (setosa) =", round(d_same_class, 3))
