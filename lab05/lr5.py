"""ЛР № 5. Линейные методы: логистическая регрессия и SVM.

Вариант 10: C_VAR = 100, KERNEL_VAR = 'rbf'.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

C_VAR = 100
KERNEL_VAR = "rbf"

# --- ШАГ 1: LogisticRegression на Iris ---
iris = load_iris()
Xi, yi = iris.data, iris.target
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    Xi, yi, test_size=0.3, stratify=yi, random_state=42,
)

logreg_iris = LogisticRegression(max_iter=1000)
logreg_iris.fit(Xi_train, yi_train)
acc_lr_iris = accuracy_score(yi_test, logreg_iris.predict(Xi_test))
print("Iris | LogReg accuracy =", round(acc_lr_iris, 4))
print("coef_.shape     =", logreg_iris.coef_.shape)
print("intercept_.shape=", logreg_iris.intercept_.shape)
print("coef_:\n", logreg_iris.coef_.round(3))
print("intercept_:", logreg_iris.intercept_.round(3))

# --- ШАГ 2: LogisticRegression с разными C на Wine ---
wine = load_wine()
Xw, yw = wine.data, wine.target
Xw_train, Xw_test, yw_train, yw_test = train_test_split(
    Xw, yw, test_size=0.3, stratify=yw, random_state=42,
)
scaler = StandardScaler()
Xw_train_s = scaler.fit_transform(Xw_train)
Xw_test_s = scaler.transform(Xw_test)

print("\nWine, LogReg с разными C (со стандартизацией):")
print("  C       accuracy  ||coef||")
for C_val in (0.01, 1, 100):
    m = LogisticRegression(C=C_val, max_iter=1000)
    m.fit(Xw_train_s, yw_train)
    acc = accuracy_score(yw_test, m.predict(Xw_test_s))
    w_norm = np.linalg.norm(m.coef_)
    print(f"  {C_val:6}  {acc:.4f}    {w_norm:.3f}")

# --- ШАГ 3: LinearSVC на Wine ---
lsvc_wine = LinearSVC(C=1.0, max_iter=5000)
lsvc_wine.fit(Xw_train_s, yw_train)
acc_lsvc_wine = accuracy_score(yw_test, lsvc_wine.predict(Xw_test_s))
print(f"\nWine | LinearSVC(C=1) accuracy = {acc_lsvc_wine:.4f}")

# --- ШАГ 4: SVC-RBF vs LinearSVC на Digits ---
digits = load_digits()
Xd, yd = digits.data, digits.target
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    Xd, yd, test_size=0.3, stratify=yd, random_state=42,
)

svc_rbf = SVC(kernel="rbf", C=1, gamma="scale")
svc_rbf.fit(Xd_train, yd_train)
acc_svc_rbf = accuracy_score(yd_test, svc_rbf.predict(Xd_test))

lsvc_dig = LinearSVC(C=1.0, max_iter=5000)
lsvc_dig.fit(Xd_train, yd_train)
acc_lsvc_dig = accuracy_score(yd_test, lsvc_dig.predict(Xd_test))

print(f"\nDigits | SVC(rbf, C=1)  = {acc_svc_rbf:.4f}")
print(f"Digits | LinearSVC(C=1) = {acc_lsvc_dig:.4f}")
print("Ядро RBF дает нелинейную границу — на 64 пикселях цифры плохо разделимы линейно.")

# --- ШАГ 5: SVC с параметрами варианта ---
svc_var = SVC(kernel=KERNEL_VAR, C=C_VAR, gamma="scale")
svc_var.fit(Xd_train, yd_train)
acc_var = accuracy_score(yd_test, svc_var.predict(Xd_test))
print(f"\nDigits | SVC(kernel={KERNEL_VAR}, C={C_VAR}) = {acc_var:.4f}")
print(f"Для сравнения, SVC(rbf, C=1) из шага 4 = {acc_svc_rbf:.4f}")

# --- ШАГ 6: решающая граница на двух признаках Iris (petal length x petal width) ---
Xi2 = Xi[:, [2, 3]]
Xi2_train, Xi2_test, yi2_train, yi2_test = train_test_split(
    Xi2, yi, test_size=0.3, stratify=yi, random_state=42,
)

model_2d = LogisticRegression(max_iter=1000)
model_2d.fit(Xi2_train, yi2_train)

x_min, x_max = Xi2[:, 0].min() - 0.5, Xi2[:, 0].max() + 0.5
y_min, y_max = Xi2[:, 1].min() - 0.5, Xi2[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300),
)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model_2d.predict(grid).reshape(xx.shape)

plt.figure(figsize=(6, 5))
plt.contourf(xx, yy, Z, alpha=0.3)
for cls in range(len(iris.target_names)):
    mask = yi2_train == cls
    plt.scatter(Xi2_train[mask, 0], Xi2_train[mask, 1],
                label=iris.target_names[cls], edgecolor="k")
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.title("Iris (petal): решающая граница LogisticRegression")
plt.legend()
plt.tight_layout()
plt.savefig("iris_decision_boundary.png", dpi=120)
plt.close()
