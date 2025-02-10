import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

# 1. Veri Setini Yükle
df = pd.read_csv("IRIS.csv")

# 2. Kategorik Değişkeni Sayısala Çevir (Label Encoding)
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# 3. Özellikleri ve Etiketleri Ayır
X = df.drop('species', axis=1)  # Bağımsız değişkenler
y = df['species']  # Bağımlı değişken (etiket)

# 4. Özellik Standardizasyonu (Z-Score Normalizasyonu)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Eğitim ve Test Setlerine Ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- Model İyileştirme Adımları ---

# 6. Regularizasyon: Ridge (L2) ve Lasso (L1) Regularizasyonu
# Ridge (L2 Regularization)
logreg_ridge = LogisticRegression(penalty='l2', max_iter=200)
logreg_ridge.fit(X_train, y_train)

# Lasso (L1 Regularization)
logreg_lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=200)
logreg_lasso.fit(X_train, y_train)

# 7. Hiperparametre Optimizasyonu: GridSearchCV ile Logistic Regression için optimize etme
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Regularizasyon katsayısı
    'solver': ['liblinear', 'saga']  # Çeşitli solverlar
}
grid_search = GridSearchCV(LogisticRegression(max_iter=200), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# En iyi hiperparametreleri yazdıralım
print("\nEn iyi hiperparametreler (Logistic Regression):")
print(grid_search.best_params_)

# 8. Alternatif Model Seçimi: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 9. Cross-Validation: Model performansını daha iyi değerlendirebilmek için çapraz doğrulama yapalım
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_logreg = cross_val_score(logreg_ridge, X_scaled, y, cv=cv, scoring='accuracy')
cv_scores_rf = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')

print(f"\nLogistic Regression Cross-validation Skorları: {cv_scores_logreg}")
print(f"\nRandom Forest Cross-validation Skorları: {cv_scores_rf}")

# 10. Model Performansı: En iyi modeli seçmek ve doğruluk oranını hesaplamak
# GridSearchCV ile en iyi modeli seçtik
best_model = grid_search.best_estimator_

# Test seti üzerinde değerlendirme
y_pred_best = best_model.predict(X_test)

# 11. Sonuçları Görüntüle
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"\nEn iyi modelin Test Doğruluk Oranı: {accuracy_best:.4f}")
print("\nSınıflandırma Raporu (En İyi Model):")
print(classification_report(y_test, y_pred_best))

# 12. Karışıklık Matrisi (Confusion Matrix)
conf_matrix_best = confusion_matrix(y_test, y_pred_best)
print("\nKarışıklık Matrisi (En İyi Model):")
print(conf_matrix_best)

# --- Visualize ROC Curve for Best Model ---
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1], pos_label=1)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
