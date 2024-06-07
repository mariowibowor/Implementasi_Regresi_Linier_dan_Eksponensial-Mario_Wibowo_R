import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Memuat dataset
data = pd.read_csv('https://drive.google.com/uc?export=download&id=1_QpqeFBe_Ok0dT-pmu-svVUPZzOMswEG')
X_TB = data['Hours Studied'].values.reshape(-1, 1)
y_NT = data['Performance Index'].values

# Regresi Linier
linear_model = LinearRegression()
linear_model.fit(X_TB, y_NT)
y_pred_linear = linear_model.predict(X_TB)

# Plotting
plt.scatter(X_TB, y_NT, color='blue', label='Data Asli')
plt.plot(X_TB, y_pred_linear, color='red', label='Regresi Linear')
plt.title('Regresi Linear untuk Durasi Waktu Belajar terhadap Nilai Ujian')
plt.xlabel('Durasi Waktu Belajar (TB)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()
plt.show()

# Menghitung RMSE
rmse_linear = np.sqrt(mean_squared_error(y_NT, y_pred_linear))
print(f'RMSE untuk Model Linear: {rmse_linear}')
