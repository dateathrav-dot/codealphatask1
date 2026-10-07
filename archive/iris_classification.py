import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix, ConfusionMatrixDisplay)

# 1. Load data -------------------------------------------------------------
path = sys.argv[1] if len(sys.argv) > 1 else "Iris.csv"
df = pd.read_csv(path)
if "Id" in df.columns:
    df = df.drop(columns="Id")          # identifier, not a feature
print("Shape:", df.shape)
print("Missing values:", int(df.isna().sum().sum()))
print(df["Species"].value_counts(), "\n")

X = df.drop(columns="Species")
y = df["Species"]

# 2. Quick exploration plot --------------------------------------------------
sns.pairplot(df, hue="Species")
plt.savefig("pairplot.png", dpi=110, bbox_inches="tight")
plt.close()

# 3. Train / test split (stratified keeps 1/3 of each class in both sets) -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Compare several models with 5-fold CV on the training set ---------------
models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=200)),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf")),
}
print("Cross-validation accuracy (training data):")
for name, m in models.items():
    s = cross_val_score(m, X_train, y_train, cv=5)
    print(f"  {name:20s} {s.mean():.3f} +/- {s.std():.3f}")

# 5. Final model: evaluate on the held-out test set ---------------------------
model = models["SVM"]
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\nTest accuracy:", round(accuracy_score(y_test, pred), 4))
print("\nClassification report:\n", classification_report(y_test, pred))
print("Confusion matrix:\n", confusion_matrix(y_test, pred))

ConfusionMatrixDisplay.from_predictions(y_test, pred, cmap="Blues", xticks_rotation=20)
plt.title("Confusion matrix (test set)")
plt.savefig("confusion_matrix.png", dpi=110, bbox_inches="tight")
plt.close()

# 6. Predict a new flower -----------------------------------------------------
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("\nPrediction for [5.1, 3.5, 1.4, 0.2]:", model.predict(sample)[0])