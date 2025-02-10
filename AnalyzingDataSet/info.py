import pandas as pd

# 1. Veri setinin yolunu belirtin
data_file_path = "IRIS.csv"  # Çıkardığınız dosyanın adı ve yolu

# 2. CSV dosyasını Pandas DataFrame olarak yükleyin
df = pd.read_csv(data_file_path)

# 3. İlk birkaç satırı görüntüleyin
print("Veri Setinin İlk 5 Satırı:")
print(df.head())

# 4. Veri setinin genel bilgilerinin alınması
print("\nVeri Seti Hakkında Genel Bilgiler:")
print(df.info())

# 5. Temel istatistiksel bilgiler
print("\nTemel İstatistiksel Bilgiler:")
print(df.describe())

# 6. Sütun isimlerini kontrol edin
print("\nSütun İsimleri:")
print(df.columns)

# 7. Eksik verilerin kontrolü
print("\nEksik Veri Kontrolü:")
print(df.isnull().sum())
