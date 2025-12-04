import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype

# -----------------------------------------------------------
# 1. Load Dataset
# -----------------------------------------------------------
df = pd.read_csv("Loan_default.csv")

# -----------------------------------------------------------
# 2. Drop ID-like columns (critical)
# -----------------------------------------------------------
id_like = [col for col in df.columns if 'id' in col.lower()]
df = df.drop(columns=id_like)

# -----------------------------------------------------------
# 3. Clean Missing Values (safe)
# -----------------------------------------------------------
for col in df.columns:
    if is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------------------------------------
# 4. Encode Categorical Columns (LabelEncode, NOT one-hot)
# -----------------------------------------------------------
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# -----------------------------------------------------------
# 5. Train/Test Split
# -----------------------------------------------------------
target = "Default"
X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 6. Standard Scaling (only numerical)
# -----------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------------------------------------
# 7. Memory-Efficient Linear SVM
# -----------------------------------------------------------
model = LinearSVC(class_weight="balanced", max_iter=5000)
model.fit(X_train, y_train)

# -----------------------------------------------------------
# 8. Evaluation
# -----------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)

# Visualization
class_labels = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

# Confusion Matrix Output
sns.heatmap(pd.DataFrame(conf_matrix), annot=True, cmap="YlGnBu", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion Matrix', y=1.1)
plt.ylabel('Actual label (Ground Truth)')
plt.xlabel('Predicted label')
plt.show()

# --- 8. Visualization of Performance Metrics ---
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
metrics_values = [accuracy, precision, recall, f1_score]

plt.figure(figsize=(9, 5))
bar_plot = plt.bar(metrics_names, metrics_values, color=['#4daf4a', '#377eb8', '#ff7f00', '#984ea3'])
plt.ylim(0.0, 1.0) # Metrics are always between 0 and 1
plt.title('Performance Metrics', fontsize=16)
plt.ylabel('Score', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

for bar in bar_plot:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
