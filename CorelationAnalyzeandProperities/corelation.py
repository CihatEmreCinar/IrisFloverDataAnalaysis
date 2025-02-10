import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Veri Setini Yükle
df = pd.read_csv("IRIS.csv")

# 2. Özellik Seçimi: Korelasyon Matrisi
# Sadece sayısal verilerle korelasyon hesaplıyoruz
correlation_matrix = df.iloc[:, :-1].corr()  # Son sütun (species) hariç
print("\nKorelasyon Matrisi:")
print(correlation_matrix)

# 3. Yüksek Korelasyona Sahip Özelliklerin Kaldırılması
# Sepal Length ve Petal Length arasında yüksek korelasyon olduğunu görebiliriz. Bu durumda, birini seçebiliriz.
# Özellikle, modelinizde gereksiz özellikleri kaldırmak, overfitting'i engellemeye yardımcı olabilir.
# Burada sepal_length'i kaldırıyoruz.
df = df.drop('sepal_length', axis=1)

# 4. Kategorik Değişken Dönüşümü: LabelEncoder
# 'species' kategorik değişkenini sayısal verilere dönüştürme
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# 5. Özellik Standardizasyonu (Z-Score Normalizasyonu)
# Özelliklerin hepsinin aynı ölçekte olması için standardizasyon yapılabilir. Bu özellikle mesafe tabanlı algoritmalar için önemlidir.
scaler = StandardScaler()
X = df.drop('species', axis=1)  # Bağımsız değişkenler
y = df['species']  # Bağımlı değişken

X_scaled = scaler.fit_transform(X)

# 6. Özellik Normalizasyonu (Min-Max Scaling)
# Veriyi 0-1 aralığına dönüştürmek için MinMaxScaler kullanıyoruz
normalizer = MinMaxScaler()
X_normalized = normalizer.fit_transform(X)

# 7. Veriyi Eğitim ve Test Setlerine Ayırma
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_normalized, X_test_normalized = train_test_split(X_normalized, test_size=0.2, random_state=42)

# 8. Sonuçları Görüntüleme
print("\nÖzellikler ve Etiketler (encoded, scaled, and normalized):")
print("Scaled Özellikler (ilk 5):")
print(X_train_scaled[:5])  # İlk 5 örnek (Scaled)
print("Normalized Özellikler (ilk 5):")
print(X_train_normalized[:5])  # İlk 5 örnek (Normalized)
print("\nEtiketler (ilk 5):")
print(y_train[:5])  # İlk 5 etiket
