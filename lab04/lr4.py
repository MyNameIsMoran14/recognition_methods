"""ЛР № 4. Байесовские классификаторы: GaussianNB, LDA, QDA.

Вариант 10: датасет для шага 4 — Wine, priors = [0.33, 0.4, 0.27].
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

DATASET_VAR = "Wine"
PRIORS_VAR = [0.33, 0.4, 0.27]

# --- ШАГ 1: GaussianNB на Iris + чтение theta_ и var_ ---
iris = load_iris()
Xi, yi = iris.data, iris.target
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.3, stratify=yi, random_state=42,
)

gnb = GaussianNB()
gnb.fit(Xi_train, yi_train)
acc_gnb_iris = accuracy_score(yi_test, gnb.predict(Xi_test))
print("Iris | GaussianNB accuracy =", round(acc_gnb_iris, 4))
print("theta_ (средние m_i, форма", gnb.theta_.shape, "):\n", gnb.theta_.round(3))
print("var_ (диагональ S_i, форма", gnb.var_.shape, "):\n", gnb.var_.round(3))

# --- ШАГ 2: сравнение GNB, LDA, QDA на Iris ---
lda_iris = LinearDiscriminantAnalysis()
lda_iris.fit(Xi_train, yi_train)
acc_lda_iris = accuracy_score(yi_test, lda_iris.predict(Xi_test))

qda_iris = QuadraticDiscriminantAnalysis()
qda_iris.fit(Xi_train, yi_train)
acc_qda_iris = accuracy_score(yi_test, qda_iris.predict(Xi_test))

print("\nСравнение на Iris:")
print(f"  GaussianNB : {acc_gnb_iris:.4f}")
print(f"  LDA        : {acc_lda_iris:.4f}")
print(f"  QDA        : {acc_qda_iris:.4f}")

# --- ШАГ 3: то же самое на Wine (со стандартизацией) ---
wine = load_wine()
Xw, yw = wine.data, wine.target
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.3, stratify=yw, random_state=42,
)
scaler = StandardScaler()
Xw_train_s = scaler.fit_transform(Xw_train)
Xw_test_s = scaler.transform(Xw_test)

acc_wine = {}
for name, model in [
    ("GaussianNB", GaussianNB()),
    ("LDA", LinearDiscriminantAnalysis()),
    ("QDA", QuadraticDiscriminantAnalysis()),
]:
    model.fit(Xw_train_s, yw_train)
    acc_wine[name] = accuracy_score(yw_test, model.predict(Xw_test_s))

print("\nСравнение на Wine (со стандартизацией):")
for name, acc in acc_wine.items():
    print(f"  {name:11s}: {acc:.4f}")

# --- ШАГ 4: влияние priors в GaussianNB (вариант: Wine) ---
Xv_train, Xv_test, yv_train, yv_test = Xw_train_s, Xw_test_s, yw_train, yw_test

gnb_default = GaussianNB()
gnb_default.fit(Xv_train, yv_train)
acc_default = accuracy_score(yv_test, gnb_default.predict(Xv_test))

gnb_priors = GaussianNB(priors=PRIORS_VAR)
gnb_priors.fit(Xv_train, yv_train)
acc_priors = accuracy_score(yv_test, gnb_priors.predict(Xv_test))

print(f"\n{DATASET_VAR} | GNB, priors=None       : {acc_default:.4f}")
print(f"{DATASET_VAR} | GNB, priors={PRIORS_VAR}: {acc_priors:.4f}")

# --- ШАГ 5: GaussianNB vs kNN на Digits ---
digits = load_digits()
Xd, yd = digits.data, digits.target
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, stratify=yd, random_state=42,
)

gnb_dig = GaussianNB()
gnb_dig.fit(Xd_train, yd_train)
yd_pred_gnb = gnb_dig.predict(Xd_test)
acc_gnb_dig = accuracy_score(yd_test, yd_pred_gnb)

knn_dig = KNeighborsClassifier(n_neighbors=3)
knn_dig.fit(Xd_train, yd_train)
acc_knn_dig = accuracy_score(yd_test, knn_dig.predict(Xd_test))

print("\nDigits | GaussianNB :", round(acc_gnb_dig, 4))
print("Digits | kNN k=3    :", round(acc_knn_dig, 4))
print("Разрыв объясняется зависимостью соседних пикселей: "
      "наивное предположение о независимости признаков нарушено.")

# --- ШАГ 6: матрица ошибок GaussianNB на Digits ---
cm = confusion_matrix(yd_test, yd_pred_gnb)
print("\nМатрица ошибок GaussianNB на Digits (10x10):\n", cm)

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    yd_test, yd_pred_gnb, ax=ax, cmap="Blues", colorbar=False,
)
ax.set_title("GaussianNB на Digits: матрица ошибок")
plt.tight_layout()
plt.savefig("digits_gnb_confmat.png", dpi=120)
plt.close()

cm_no_diag = cm.copy()
np.fill_diagonal(cm_no_diag, 0)
idx_sorted = np.argsort(cm_no_diag.flatten())[::-1]
print("Топ-3 пар путаемых цифр (истинное -> предсказанное):")
for k in range(3):
    i, j = divmod(idx_sorted[k], 10)
    print(f"  {i} -> {j}: {cm[i, j]} раз")
