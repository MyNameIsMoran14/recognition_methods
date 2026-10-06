"""ЛР № 3. Метод k ближайших соседей.

Вариант 10: K = 1 и 15 на Digits, метрика manhattan.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

K_VARIANTS = (1, 15)   # K: оба значения из варианта 10
METRIC_VAR = "manhattan"

# --- ШАГ 1: kNN на Wine БЕЗ масштабирования ---
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42,
)

knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))
print("Wine БЕЗ StandardScaler, k=5: accuracy =", round(acc_raw, 4))

# --- ШАГ 2: kNN на Wine СО StandardScaler ---
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn_std = KNeighborsClassifier(n_neighbors=5)
knn_std.fit(X_train_s, y_train)
acc_std = accuracy_score(y_test, knn_std.predict(X_test_s))
print("Wine СО StandardScaler, k=5: accuracy =", round(acc_std, 4))
print("Разница (StandardScaler - без):", round(acc_std - acc_raw, 4))

# --- ШАГ 3: зависимость accuracy от k на стандартизованном Wine ---
print("\nЗависимость accuracy от k на Wine (со стандартизацией):")
print("  k   accuracy")
for k in (1, 3, 5, 7, 15):
    m = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train_s, y_train)
    acc_k = accuracy_score(y_test, m.predict(X_test_s))
    print(f"  {k:2d}  {acc_k:.4f}")

# --- ШАГ 4: kNN на Digits с параметрами варианта (оба значения K) ---
digits = load_digits()
Xd, yd = digits.data, digits.target
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, stratify=yd, random_state=42,
)

print(f"\nDigits, metric={METRIC_VAR}:")
acc_by_k_manhattan = {}
for k in K_VARIANTS:
    m = KNeighborsClassifier(n_neighbors=k, metric=METRIC_VAR)
    m.fit(Xd_train, yd_train)
    acc = accuracy_score(yd_test, m.predict(Xd_test))
    acc_by_k_manhattan[k] = acc
    print(f"  k={k:2d}: accuracy = {acc:.4f}")

# --- ШАГ 5: визуализация 8 ошибочно классифицированных цифр (k=1, manhattan) ---
knn_errors_model = KNeighborsClassifier(n_neighbors=K_VARIANTS[0], metric=METRIC_VAR)
knn_errors_model.fit(Xd_train, yd_train)
yd_pred = knn_errors_model.predict(Xd_test)

errors = np.where(yd_pred != yd_test)[0]
print(f"\nЧисло ошибок на тесте Digits (k={K_VARIANTS[0]}, {METRIC_VAR}):", len(errors))

n_show = min(8, len(errors))
fig, axes = plt.subplots(2, 4, figsize=(8, 4))
for i in range(n_show):
    idx = errors[i]
    img = Xd_test[idx].reshape(8, 8)
    ax = axes[i // 4, i % 4]
    ax.imshow(img, cmap="gray")
    ax.set_title(f"пред. {yd_pred[idx]}, ист. {yd_test[idx]}", fontsize=9)
    ax.axis("off")
for j in range(n_show, 8):
    axes[j // 4, j % 4].axis("off")
plt.tight_layout()
plt.savefig("digits_errors.png", dpi=120)
plt.close()

# --- ШАГ 6: сравнение с другой метрикой (euclidean) при тех же K ---
other_metric = "euclidean"
print(f"\nСравнение метрик на Digits ({METRIC_VAR} vs {other_metric}):")
print("  k   manhattan   euclidean   разница")
for k in K_VARIANTS:
    m = KNeighborsClassifier(n_neighbors=k, metric=other_metric)
    m.fit(Xd_train, yd_train)
    acc_other = accuracy_score(yd_test, m.predict(Xd_test))
    acc_manh = acc_by_k_manhattan[k]
    print(f"  {k:2d}  {acc_manh:.4f}      {acc_other:.4f}      {abs(acc_manh - acc_other):.4f}")
