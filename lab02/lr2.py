"""ЛР № 2. Признаки, метрики, стандартизация и разбиение выборки.

Вариант 10: пара признаков Wine для диаграммы рассеяния —
0 (alcohol), 6 (flavanoids).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.3
FEATURE_1 = 0  # alcohol (вариант 10)
FEATURE_2 = 6  # flavanoids (вариант 10)

# --- ШАГ 1: три расстояния вручную между двумя цветками Iris ---
iris = load_iris()
X_iris = iris.data
x1 = X_iris[0]
x2 = X_iris[100]

d2_manual = np.sqrt(np.sum((x1 - x2) ** 2))
d1_manual = np.sum(np.abs(x1 - x2))
dinf_manual = np.max(np.abs(x1 - x2))
print("Вручную: d_2 =", round(d2_manual, 3), " d_1 =", round(d1_manual, 3), " d_inf =", round(dinf_manual, 3))

# --- ШАГ 2: те же три расстояния через sklearn ---
x1_row = x1.reshape(1, -1)
x2_row = x2.reshape(1, -1)
for metric in ("euclidean", "manhattan", "chebyshev"):
    d_sklearn = pairwise_distances(x1_row, x2_row, metric=metric)[0, 0]
    print(f"sklearn [{metric}] =", round(d_sklearn, 3))

# --- ШАГ 3: масштабы признаков Wine ---
wine = load_wine()
X = wine.data
y = wine.target

print("Средние по признакам Wine:\n", X.mean(axis=0).round(3))
print("СКО по признакам Wine:\n", X.std(axis=0).round(3))

# --- ШАГ 4: стандартизация и нормализация ---
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X)
print("mean_ =", std_scaler.mean_.round(3))
print("scale_ =", std_scaler.scale_.round(3))
print("Проверка mean(X_std) ~ 0:", X_std.mean(axis=0).round(3))
print("Проверка std(X_std) ~ 1:", X_std.std(axis=0).round(3))

mm_scaler = MinMaxScaler()
X_mm = mm_scaler.fit_transform(X)
print("data_min_ =", mm_scaler.data_min_.round(3))
print("data_max_ =", mm_scaler.data_max_.round(3))
print("Проверка min(X_mm) = 0:", X_mm.min(axis=0))
print("Проверка max(X_mm) = 1:", X_mm.max(axis=0))

# --- ШАГ 5: боксплоты и диаграммы рассеяния до/после стандартизации ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(X, showfliers=False)
axes[0].set_title("Wine: исходные признаки")
axes[0].set_xlabel("номер признака")
axes[1].boxplot(X_std, showfliers=False)
axes[1].set_title("Wine: после StandardScaler")
axes[1].set_xlabel("номер признака")
plt.tight_layout()
plt.savefig("wine_boxplots.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for cls in range(len(wine.target_names)):
    mask = y == cls
    axes[0].scatter(X[mask, FEATURE_1], X[mask, FEATURE_2], label=f"класс {cls}")
    axes[1].scatter(X_std[mask, FEATURE_1], X_std[mask, FEATURE_2], label=f"класс {cls}")
axes[0].set_xlabel(wine.feature_names[FEATURE_1])
axes[0].set_ylabel(wine.feature_names[FEATURE_2])
axes[0].set_title(f"До: {wine.feature_names[FEATURE_1]} и {wine.feature_names[FEATURE_2]}")
axes[0].legend()
axes[1].set_xlabel(wine.feature_names[FEATURE_1])
axes[1].set_ylabel(wine.feature_names[FEATURE_2])
axes[1].set_title("После стандартизации")
axes[1].legend()
plt.tight_layout()
plt.savefig("wine_scatter_before_after.png", dpi=120)
plt.close()

# --- ШАГ 6: стратифицированное разбиение выборки ---
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)
print("Размер X_train:", X_train.shape, " X_test:", X_test.shape)
print("bincount(y_train) =", np.bincount(y_train))
print("bincount(y_test) =", np.bincount(y_test))

train_fracs = np.bincount(y_train) / len(y_train)
test_fracs = np.bincount(y_test) / len(y_test)
print("Доли классов в train:", train_fracs.round(3))
print("Доли классов в test:", test_fracs.round(3))
