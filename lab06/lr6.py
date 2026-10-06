"""ЛР № 6. Оценка качества и снижение размерности. Итоговое сравнение.

Вариант 10: N_COMP_VAR = 30, MODEL_FINAL_VAR = 'lda'.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

N_COMP_VAR = 30
MODEL_FINAL_VAR = "lda"

# --- ШАГ 1: фиксированное разбиение vs кросс-валидация на Iris ---
iris = load_iris()
Xi, yi = iris.data, iris.target
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.3, stratify=yi, random_state=42,
)

lr_iris = LogisticRegression(max_iter=1000)
lr_iris.fit(Xi_train, yi_train)
acc_fixed = accuracy_score(yi_test, lr_iris.predict(Xi_test))
print("Iris | LogReg fixed test =", round(acc_fixed, 4))

cv_scores = cross_val_score(
    LogisticRegression(max_iter=1000), Xi, yi, cv=5, scoring="accuracy",
)
print("Iris | LogReg CV 5-fold  =", cv_scores.round(4))
print("  среднее =", round(cv_scores.mean(), 4), " СКО =", round(cv_scores.std(), 4))

# --- ШАГ 2: classification_report и матрица ошибок на Digits ---
digits = load_digits()
Xd, yd = digits.data, digits.target
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, stratify=yd, random_state=42,
)

knn_dig = KNeighborsClassifier(n_neighbors=3)
knn_dig.fit(Xd_train, yd_train)
yd_pred = knn_dig.predict(Xd_test)

print("\nclassification_report на Digits (kNN, k=3):")
print(classification_report(yd_test, yd_pred, digits=3))

fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay.from_predictions(
    yd_test, yd_pred, ax=ax, cmap="Blues", colorbar=False,
)
ax.set_title("kNN(k=3) на Digits: матрица ошибок")
plt.tight_layout()
plt.savefig("digits_knn_confmat.png", dpi=120)
plt.close()

# --- ШАГ 3: PCA на Digits — scatter и кривая объясненной дисперсии ---
pca2 = PCA(n_components=2)
Xd_2d = pca2.fit_transform(Xd)

plt.figure(figsize=(7, 6))
for cls in range(10):
    mask = yd == cls
    plt.scatter(Xd_2d[mask, 0], Xd_2d[mask, 1], s=10, label=str(cls))
plt.xlabel("PCA-1")
plt.ylabel("PCA-2")
plt.title("Digits: проекция на первые 2 главные компоненты")
plt.legend(markerscale=2, ncol=2, fontsize=9)
plt.tight_layout()
plt.savefig("digits_pca2.png", dpi=120)
plt.close()

pca_full = PCA(n_components=N_COMP_VAR)
pca_full.fit(Xd)
cum = pca_full.explained_variance_ratio_.cumsum()
print("\nНакопленная объясненная дисперсия по компонентам (первые и каждая 5-я):")
for k, v in enumerate(cum, start=1):
    if k <= 3 or k % 5 == 0:
        print(f"  {k:2d} компонент: {v:.4f}")
n_90 = int(np.argmax(cum >= 0.90)) + 1 if (cum >= 0.90).any() else N_COMP_VAR + 1
print(f"Первое k, при котором доля >= 0.90: {n_90}")

# --- ШАГ 4: kNN на исходных 64 признаках vs на 10 главных компонентах ---
def time_knn(Xtr, Xte, ytr, yte):
    t0 = time.perf_counter()
    m = KNeighborsClassifier(n_neighbors=3)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    t1 = time.perf_counter()
    return accuracy_score(yte, yp), t1 - t0


acc_full, t_full = time_knn(Xd_train, Xd_test, yd_train, yd_test)

pca10 = PCA(n_components=10)
Xd_train_10 = pca10.fit_transform(Xd_train)
Xd_test_10 = pca10.transform(Xd_test)
acc_pca, t_pca = time_knn(Xd_train_10, Xd_test_10, yd_train, yd_test)

print("\nDigits: kNN на исходных vs PCA-сжатых признаках")
print("  признаков  accuracy  время (с)")
print(f"      64     {acc_full:.4f}   {t_full:.4f}")
print(f"      10     {acc_pca:.4f}   {t_pca:.4f}")

# --- ШАГ 5: итоговая сводная таблица 3 x 5 ---
def prepare(dataset_name):
    if dataset_name == "Iris":
        d = load_iris()
        scale = False
    elif dataset_name == "Wine":
        d = load_wine()
        scale = True
    else:
        d = load_digits()
        scale = False
    X, y = d.data, d.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42,
    )
    if scale:
        sc = StandardScaler()
        Xtr = sc.fit_transform(Xtr)
        Xte = sc.transform(Xte)
    return Xtr, Xte, ytr, yte


def build_model(name):
    return {
        "knn": KNeighborsClassifier(n_neighbors=3),
        "gnb": GaussianNB(),
        "lda": LinearDiscriminantAnalysis(),
        "logreg": LogisticRegression(max_iter=1000),
        "svc_rbf": SVC(kernel="rbf", C=1, gamma="scale"),
    }[name]


datasets = ["Iris", "Wine", "Digits"]
models = ["knn", "gnb", "lda", "logreg", "svc_rbf"]
results = np.zeros((len(datasets), len(models)))

for i, dname in enumerate(datasets):
    Xtr, Xte, ytr, yte = prepare(dname)
    for j, mname in enumerate(models):
        m = build_model(mname)
        m.fit(Xtr, ytr)
        results[i, j] = accuracy_score(yte, m.predict(Xte))

print("\nИтоговая сводная таблица (accuracy на фиксированном тесте):")
print(f"{'':10s}" + "".join(f" {n:>8s}" for n in models))
for i, dname in enumerate(datasets):
    row = "".join(f" {v:8.4f}" for v in results[i])
    print(f"{dname:10s}{row}")
    best_j = int(np.argmax(results[i]))
    print(f"            лучшая: {models[best_j]}")

print(f"\n5-фолдовая CV для модели '{MODEL_FINAL_VAR}':")
for dname in datasets:
    Xtr, Xte, ytr, yte = prepare(dname)
    scores = cross_val_score(build_model(MODEL_FINAL_VAR), Xtr, ytr, cv=5, scoring="accuracy")
    print(f"  {dname:6s}: mean = {scores.mean():.4f}, std = {scores.std():.4f}")
