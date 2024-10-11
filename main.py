import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



# Verileri bir sözlük şeklinde tanımlıyoruz
data = {
    'Deneyim Yılı (x)': [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
    'Maaş (y)': [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
}

# Verileri bir DataFrame'e dönüştürüyoruz
df = pd.DataFrame(data)

# DataFrame'i görüntüleme
print(df)

type(df)

# 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz. Bias = 275, Weight= 90 (y’ = b+wx)

# y_hat = 275 + (90 * x)

# 2- Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.

# Girdileri (X) ve çıktıları (y) belirle
X = df[['Deneyim Yılı (x)']]  # Girdiler (Deneyim Yılı)
y = df['Maaş (y)']  # Çıktılar (Maaş)

# Doğrusal regresyon modelini oluştur ve eğit
model = LinearRegression()
model.fit(X, y)

# Modelin katsayılarını yazdır
print(f"Y-intercept (b0): {model.intercept_}")
print(f"Slope (b1): {model.coef_[0]}")

# Tahmin edilen maaşlar
y_pred = model.predict(X)

df["y_pred"] = y_pred

df


# Gerçek ve tahmin edilen verileri görselleştir
plt.scatter(X, y, color='blue', label='Gerçek Veriler')
plt.plot(X, y_pred, color='red', label='Model Tahminleri')
plt.xlabel('Deneyim Yılı (x)')
plt.ylabel('Maaş (y)')
plt.title('Basit Doğrusal Regresyon')
plt.legend()
plt.show()

# 3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız

# MSE, RMSE, MAE hesaplama
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)

# Sonuçları yazdır
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")


# MSE, RMSE, MAE'nin veri setine eklenmesi
df["MSE"] = mse
df["RMSE"] = rmse
df["MAE"] = mae

df

# MSE, RMSE, MAE'yi manuel olarak hesaplama

n = len(y)  # Veri sayısı

# Gerçek değerler ile tahmin edilen değerler arasındaki farkları hesapla
errors = y - y_pred

# MSE hesaplama
mse_manual = (errors ** 2).sum() / n

# RMSE hesaplama
rmse_manual = np.sqrt(mse_manual)

# MAE hesaplama
mae_manual = np.abs(errors).sum() / n

# Sonuçları yazdır
print(f"Manuel Mean Squared Error (MSE): {mse_manual}")
print(f"Mean Squared Error (MSE): {mse}")

print(f"Manuel Root Mean Squared Error (RMSE): {rmse_manual}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

print(f"Manuel Mean Absolute Error (MAE): {mae_manual}")
print(f"Mean Absolute Error (MAE): {mae}")

# MSE_MANUAL, RMSE_MANUAL ve MAE_MANUAL'in veri setine eklenmesi
df["MSE_MANUAL"] = mse_manual
df["RMSE_MANUAL"] = rmse_manual
df["MAE_MANUAL"] = mae_manual

df

# Projenin Fonksiyonlaştırılması

# Veriyi oluşturan fonksiyon
def create_dataframe():
    data = {
        'Deneyim Yılı (x)': [5, 7, 3, 3, 2, 7, 3, 10, 6, 4, 8, 1, 1, 9, 1],
        'Maaş (y)': [600, 900, 550, 500, 400, 950, 540, 1200, 900, 550, 1100, 460, 400, 1000, 380]
    }
    df = pd.DataFrame(data)
    return df

# Doğrusal regresyon modelini oluşturan ve tahminler yapan fonksiyon
def linear_regression_model(df):
    X = df[['Deneyim Yılı (x)']]  # Girdiler
    y = df['Maaş (y)']  # Çıktılar

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    df['y_pred'] = y_pred
    return model, df

# MSE, RMSE, MAE hesaplayan fonksiyon (otomatik)
def calculate_metrics_auto(y, y_pred):
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    return mse, rmse, mae

# MSE, RMSE, MAE hesaplayan fonksiyon (manuel)
def calculate_metrics_manual(y, y_pred):
    n = len(y)
    errors = y - y_pred

    mse_manual = (errors ** 2).sum() / n
    rmse_manual = np.sqrt(mse_manual)
    mae_manual = np.abs(errors).sum() / n
    return mse_manual, rmse_manual, mae_manual

# Grafik çizen fonksiyon
def plot_results(X, y, y_pred):
    plt.scatter(X, y, color='blue', label='Gerçek Veriler')
    plt.plot(X, y_pred, color='red', label='Model Tahminleri')
    plt.xlabel('Deneyim Yılı (x)')
    plt.ylabel('Maaş (y)')
    plt.title('Basit Doğrusal Regresyon')
    plt.legend()
    plt.show()

# Tüm işlemleri yürüten ana fonksiyon
def main():
    # Veriyi oluştur
    df = create_dataframe()

    # Model oluştur ve tahminler yap
    model, df = linear_regression_model(df)
    print(f"Y-intercept (b0): {model.intercept_}")
    print(f"Slope (b1): {model.coef_[0]}")

    # Otomatik metrikleri hesapla
    mse, rmse, mae = calculate_metrics_auto(df['Maaş (y)'], df['y_pred'])
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Manuel metrikleri hesapla
    mse_manual, rmse_manual, mae_manual = calculate_metrics_manual(df['Maaş (y)'], df['y_pred'])
    print(f"Manuel Mean Squared Error (MSE): {mse_manual}")
    print(f"Manuel Root Mean Squared Error (RMSE): {rmse_manual}")
    print(f"Manuel Mean Absolute Error (MAE): {mae_manual}")

    # Metrikleri DataFrame'e ekle
    df['MSE'] = mse
    df['RMSE'] = rmse
    df['MAE'] = mae
    df['MSE_MANUAL'] = mse_manual
    df['RMSE_MANUAL'] = rmse_manual
    df['MAE_MANUAL'] = mae_manual

    # Sonuçları görselleştir
    plot_results(df[['Deneyim Yılı (x)']], df['Maaş (y)'], df['y_pred'])

    # DataFrame'i döndür
    return df

# Ana fonksiyonu çalıştır
df_results = main()
print(df_results)

