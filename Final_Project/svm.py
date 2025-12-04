import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
from cuml.svm import SVC
from cuml.preprocessing import LabelEncoder
from cuml.metrics import confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd

# Load into CPU dataframe, then convert to GPU
df = pd.read_csv("Loan_default.csv")
gdf = cudf.DataFrame.from_pandas(df)

# Drop ID-like columns
id_cols = [c for c in gdf.columns if "id" in c.lower()]
gdf = gdf.drop(columns=id_cols)

# Fill missing values on GPU
for col in gdf.columns:
    if gdf[col].dtype in ["float64", "float32", "int64", "int32"]:
        gdf[col] = gdf[col].fillna(gdf[col].median())
    else:
        gdf[col] = gdf[col].fillna(gdf[col].mode()[0])

# Encode categorical columns
for col in gdf.columns:
    if gdf[col].dtype == "object":
        le = LabelEncoder()
        gdf[col] = le.fit_transform(gdf[col])

# Target
target = "Default"
X = gdf.drop(columns=[target])
y = gdf[target]

# Train/test split on GPU
train_frac = 0.75
train_size = int(len(gdf) * train_frac)

X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test  = y.iloc[train_size:]

# Standardize on GPU
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Compute non-zero variance feature count
var = X_train.var()
n_valid = int((var > 1e-12).sum())

if n_valid > 1:
    pca_components = min(50, n_valid)  # cap at 50 for speed
    print(f"Running PCA with {pca_components} components")

    pca = PCA(n_components=pca_components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

else:
    print("Skipping PCA: insufficient valid variance")

# GPU SVM (RBF kernel)
model = SVC(kernel="rbf", C=10, gamma="scale")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Convert to cuDF Series for metrics
y_pred = y_pred.astype("int32")
y_test = y_test.astype("int32")

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
# Plot confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(cm.to_array().get(), cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.colorbar()
plt.xticks(cp.arange(2), ['No Default', 'Default'])
plt.yticks(cp.arange(2), ['No Default', 'Default'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, int(cm[i, j].get()), ha='center', va='center', color='red')
plt.show()

accuracy = (y_pred == y_test).sum() / len(y_test)
print(f"Accuracy: {accuracy.get() * 100:.2f}%")

