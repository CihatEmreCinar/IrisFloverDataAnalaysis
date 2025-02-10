import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleme
df = pd.read_csv("IRIS.csv")

# 1. Veri Dağılımını Görselleştirme (Histogram)
df.iloc[:, :-1].hist(bins=20, figsize=(10, 7))
plt.suptitle("Feature Distribution (Histogram)")
plt.show()

# 2. Kategorik Değişkenin Dağılımını Görselleştirme (Box Plot)
plt.figure(figsize=(10, 6))
sns.boxplot(x='species', y='sepal_length', data=df)
plt.title("Box Plot: Sepal Length by Species")
plt.show()

# 3. Veri İlişkilerini Görselleştirme (Pairplot)
sns.pairplot(df, hue="species", markers=["o", "s", "D"], palette="Set1")
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()

# 4. Korelasyonları Görselleştirme (Heatmap)
plt.figure(figsize=(8, 6))
correlation_matrix = df.iloc[:, :-1].corr()  # Son sütun etiket (species) hariç
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# 5. Kategorik Değişkenin Dağılımını Görselleştirme (Bar Plot)
plt.figure(figsize=(8, 5))
sns.countplot(x='species', data=df, palette="Set2")
plt.title("Species Count")
plt.show()
