import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Memuat dataset
data = pd.read_csv('https://drive.google.com/uc?export=download&id=1_QpqeFBe_Ok0dT-pmu-svVUPZzOMswEG')
X_TB = data['Hours Studied'].values
y_NT = data['Performance Index'].values

# Definisikan fungsi eksponensial
def exponential_function(x, a, b):
    return a * np.exp(b * x)

# Cari parameter optimal untuk fungsi eksponensial
params, covariance = curve_fit(exponential_function, X_TB, y_NT, p0=(1, 0.1))

# Gunakan parameter optimal untuk membuat prediksi
y_pred_exponential = exponential_function(X_TB, *params)

# Plotting
plt.scatter(X_TB, y_NT, color='blue', label='Data Asli')
plt.plot(X_TB, y_pred_exponential, color='red', label='Regresi Eksponensial')
plt.title('Regresi Eksponensial untuk Durasi Waktu Belajar terhadap Nilai Ujian')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.show()

# Menghitung RMSE
rmse_exponential = np.sqrt(mean_squared_error(y_NT, y_pred_exponential))
print(f'RMSE untuk Model Eksponensial: {rmse_exponential}')
