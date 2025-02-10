import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

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

# 6. Logistic Regression Modelini Seç ve Eğit
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train, y_train)

# 7. Modeli Test Et ve Performansını Değerlendir
y_pred = logreg_model.predict(X_test)

# 8. Doğruluk Oranı ve Sınıflandırma Raporu
accuracy = accuracy_score(y_test, y_pred)
print(f"\nLogistic Regression Modeli Doğruluk Oranı: {accuracy:.4f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

# 9. Karışıklık Matrisi (Confusion Matrix)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nKarışıklık Matrisi:")
print(conf_matrix)

# 10. ROC-AUC ve ROC Eğrisi
fpr, tpr, thresholds = roc_curve(y_test, logreg_model.predict_proba(X_test)[:,1], pos_label=1)
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
