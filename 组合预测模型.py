import math
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def reverse_min_max_scaling(scaled, min_value, max_value):
    return scaled * (max_value - min_value) + min_value

def create_bigru_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(32, input_shape=input_shape)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_bilstm_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, input_shape=input_shape)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model



seed_value = 109  #双向机器学习模型必须设置全局种子不然无法复现
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)
    

excel_file1 = 'E:/预测返修/湖北高频.xlsx'
excel_file2 = 'E:/预测返修/湖北低频.xlsx'
excel_file3 = 'E:/预测返修/湖北趋势.xlsx'
excel_file_true = 'E:/预测返修/湖北真实碳价.xlsx'

    
# 读取高频数据
df_high = pd.read_excel(excel_file1)
feature_columns_high = df_high.columns.tolist()
X_high = df_high[feature_columns_high].values
y_high = df_high['price'].values
X_high = X_high.reshape((X_high.shape[0], X_high.shape[1], 1))
X_train_high, X_test_high, y_train_high, y_test_high = train_test_split(X_high, y_high, test_size=0.2, shuffle=False)
    
# 构建和训练BiGRU模型
bigru_model = create_bigru_model((X_train_high.shape[1], X_train_high.shape[2]))
bigru_model.fit(X_train_high, y_train_high, epochs=30, batch_size=32, verbose=0)
y_pred_test_high = bigru_model.predict(X_test_high).flatten()
y_pred_test_high1 = reverse_min_max_scaling(y_pred_test_high, -11.14880242, 14.72726812)
    
# 读取低频数据
df_low = pd.read_excel(excel_file2)
feature_columns_low = df_low.columns.tolist()
X_low = df_low[feature_columns_low].values
y_low = df_low['price'].values
X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, y_low, test_size=0.2, shuffle=False)
    
# 构建和训练XGBoost模型
xgb_model = XGBRegressor(subsample=0.8, reg_lambda=1, reg_alpha=0.001, n_estimators=30, min_child_weight=3, max_depth=2, learning_rate=0.2, gamma=0, colsample_bytree=0.9, random_state=seed_value)
xgb_model.fit(X_train_low, y_train_low)
y_pred_test_low = xgb_model.predict(X_test_low)
y_pred_test_low1 = reverse_min_max_scaling(y_pred_test_low, -10.75314895, 19.75295276)
    
# 读取趋势数据
df_trend = pd.read_excel(excel_file3)
feature_columns_trend = df_trend.columns.tolist()
X_trend = df_trend[feature_columns_trend].values
y_trend = df_trend['price'].values
X_trend = X_trend.reshape((X_trend.shape[0], X_trend.shape[1], 1))
X_train_trend, X_test_trend, y_train_trend, y_test_trend = train_test_split(X_trend, y_trend, test_size=0.2, shuffle=False)
    
# 构建和训练BiLSTM模型
bilstm_model = create_bilstm_model((X_train_trend.shape[1], X_train_trend.shape[2]))
bilstm_model.fit(X_train_trend, y_train_trend, epochs=30, batch_size=32, verbose=0)
y_pred_test_trend = bilstm_model.predict(X_test_trend).flatten()
y_pred_test_trend1 = reverse_min_max_scaling(y_pred_test_trend, 19.2188066, 31.57955405)
    
# 计算反归一化后的预测值总和
y_pred_test = y_pred_test_high1 + y_pred_test_low1 + y_pred_test_trend1
    
# 导入真实碳价数据
df_true = pd.read_excel(excel_file_true)
y_true = df_true.iloc[:, 0].values  # 假设真实值在第一列
    
# 计算拟合指标
mae = mean_absolute_error(y_true, y_pred_test)
mse = mean_squared_error(y_true, y_pred_test)
rmse = math.sqrt(mse)
r2 = r2_score(y_true, y_pred_test)
mape = mean_absolute_percentage_error(y_true, y_pred_test)
    
print("MAE Test:", mae)
print("RMSE Test:", rmse)
print("R2 Test:", r2)
print("MAPE Test:", mape)

