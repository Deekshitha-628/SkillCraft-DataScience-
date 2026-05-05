# SkillCraft Technology - Task 03
# Decision Tree Classifier - Bank Marketing Dataset
# Dataset: https://archive.ics.uci.edu/dataset/222/bank+marketing

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import LabelEncoder

plt.rcParams.update({
    'figure.facecolor':'#1a1a2e','axes.facecolor':'#16213e',
    'axes.labelcolor':'white','xtick.color':'white',
    'ytick.color':'white','text.color':'white',
    'grid.color':'#2a2a4a','axes.titlesize':13,
    'axes.titleweight':'bold','axes.titlecolor':'white'
})
COLORS = ['#5DCAA5','#EF9F27','#378ADD','#D4537E','#7F77DD']

# ── Load Data ─────────────────────────────────────────────────────────────────
print("Loading Bank Marketing dataset...")
df = pd.read_csv('bank-full.csv', sep=';')
df.columns = df.columns.str.strip()
print(f"✅ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"   Target (y) distribution:\n{df['y'].value_counts()}")

# ── Encode Categorical Columns ────────────────────────────────────────────────
le = LabelEncoder()
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print(f"\n✅ Encoded {len(cat_cols)} categorical columns")

# ── Features & Target ─────────────────────────────────────────────────────────
X = df.drop('y', axis=1)
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# ── Train Decision Tree ───────────────────────────────────────────────────────
dt = DecisionTreeClassifier(max_depth=5, random_state=42,
                             min_samples_split=50)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred) * 100
print(f"\n✅ Model Accuracy: {acc:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Purchase','Purchase']))

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Decision Tree Visualization
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(24, 10))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
plot_tree(dt, feature_names=X.columns.tolist(),
          class_names=['No','Yes'], filled=True,
          rounded=True, fontsize=9, ax=ax,
          impurity=False, proportion=False)
ax.set_title('Decision Tree — Bank Marketing (max_depth=5)',
             fontsize=16, fontweight='bold', color='white', pad=20)
plt.tight_layout()
plt.savefig('chart1_decision_tree.png', dpi=120,
            bbox_inches='tight', facecolor='#1a1a2e')
print("\n✅ Saved: chart1_decision_tree.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#1a1a2e')

importances = pd.Series(dt.feature_importances_,
                         index=X.columns).sort_values(ascending=True)
top10 = importances.tail(10)
colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top10)))
bars = ax.barh(top10.index, top10.values,
               color=colors, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, top10.values):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=10,
            fontweight='bold', color='white')
ax.set_title('Top 10 Feature Importances — Decision Tree')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('chart2_feature_importance.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Saved: chart2_feature_importance.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Confusion Matrix
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            linewidths=1, linecolor='white',
            xticklabels=['No Purchase','Purchase'],
            yticklabels=['No Purchase','Purchase'],
            annot_kws={'size':14,'weight':'bold'})
ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('chart3_confusion_matrix.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Saved: chart3_confusion_matrix.png")
plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Accuracy vs Tree Depth
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#1a1a2e')

depths = range(1, 16)
train_acc, test_acc = [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=42)
    m.fit(X_train, y_train)
    train_acc.append(accuracy_score(y_train, m.predict(X_train))*100)
    test_acc.append(accuracy_score(y_test,  m.predict(X_test))*100)

ax.plot(depths, train_acc, color='#5DCAA5', linewidth=2,
        marker='o', markersize=5, label='Train Accuracy')
ax.plot(depths, test_acc,  color='#EF9F27', linewidth=2,
        marker='s', markersize=5, label='Test Accuracy')
ax.axvline(x=5, color='#D4537E', linestyle='--',
           linewidth=1.5, label='Selected depth=5')
ax.set_title('Accuracy vs Tree Depth')
ax.set_xlabel('Max Depth')
ax.set_ylabel('Accuracy (%)')
ax.legend(facecolor='#1a1a2e', labelcolor='white')
ax.set_xticks(list(depths))
plt.tight_layout()
plt.savefig('chart4_accuracy_vs_depth.png', dpi=150,
            bbox_inches='tight', facecolor='#1a1a2e')
print("✅ Saved: chart4_accuracy_vs_depth.png")
plt.show()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n── Summary ──────────────────────────────────")
print(f"Total records     : {len(df)}")
print(f"Features used     : {X.shape[1]}")
print(f"Model Accuracy    : {acc:.2f}%")
print(f"Tree Depth        : 5")
print(f"Most imp. feature : {importances.idxmax()}")
print("─────────────────────────────────────────────")
print("All 4 charts saved! Upload to GitHub & LinkedIn.")