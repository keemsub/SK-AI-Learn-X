import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

TRAIN_FILE = "/mnt/elice/dataset/train.csv"
TEST_FILE = "/mnt/elice/dataset/test.csv"

data = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)
data['날짜'] = pd.to_datetime(data['날짜'], format="%Y%m%d")
data = data.set_index('날짜')

# 로그 변환 및 차분
data['price_log'] = np.log(data['철근 고장력 HD10mm 현물KRW/ton'])
data['price_log_diff'] = data['price_log'].diff().dropna()

# 시계열 특성 생성
def create_features(df, label=None):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week
    df['lag1'] = df['price_log_diff'].shift(1)
    df['lag4'] = df['price_log_diff'].shift(4)
    df['rolling_mean_4'] = df['price_log_diff'].rolling(window=4).mean()
    df['rolling_std_4'] = df['price_log_diff'].rolling(window=4).std()
    
    if label:
        df[label] = df[label]
    
    return df

data = create_features(data, label='price_log_diff')
data = data.dropna()

# 훈련 및 테스트 데이터 분할
X = data[['year', 'month', 'week', 'lag1', 'lag4', 'rolling_mean_4', 'rolling_std_4']]
y = data['price_log_diff']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost 모델 학습
model = XGBRegressor(objective='reg:squarederror', n_estimators=250, learning_rate=0.01)
model.fit(X_train_scaled, y_train)

# 예측
y_pred = model.predict(X_test_scaled)

# 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.title('Rebar Prices Forecast')
plt.xlabel('Date')
plt.ylabel('Log Differenced Price')
plt.legend()
plt.show()

# 로그 역변환
y_test_cumsum = y_test.cumsum()
y_pred_cumsum = pd.Series(y_pred, index=y_test.index).cumsum()

y_test_reversed = np.exp(y_test_cumsum + np.log(data['철근 고장력 HD10mm 현물KRW/ton'].iloc[0]))
y_pred_reversed = np.exp(y_pred_cumsum + np.log(data['철근 고장력 HD10mm 현물KRW/ton'].iloc[0]))

# 원래 가격으로 변환한 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(y_test_reversed.index, y_test_reversed, label='Actual')
plt.plot(y_pred_reversed.index, y_pred_reversed, label='Predicted')
plt.title('Rebar Prices Forecast (Original Scale)')
plt.xlabel('Date')
plt.ylabel('철근 고장력 HD10mm 현물KRW/ton')
plt.legend()
plt.show()

# 미래 48주 예측 (1주 단위)
last_index = data.index[-1]
# future_dates = pd.date_range(start=last_index + pd.Timedelta(weeks=1), periods=48, freq='W-MON')
future_dates = pd.date_range(start='2022-11-21', periods=48, freq='W-MON')

future_data = pd.DataFrame(index=future_dates)

# 마지막 데이터에서 초기값 설정
last_values = data.iloc[-1]
for col in ['lag1', 'lag4', 'rolling_mean_4', 'rolling_std_4']:
    future_data[col] = last_values[col]

# 미래 데이터 특징 생성
future_data['year'] = future_data.index.year
future_data['month'] = future_data.index.month
future_data['week'] = future_data.index.isocalendar().week

# 순차적으로 미래 데이터 예측
future_predictions = []

for i in range(48):
    future_row = future_data.iloc[i]
    future_features = scaler.transform(future_row[['year', 'month', 'week', 'lag1', 'lag4', 'rolling_mean_4', 'rolling_std_4']].values.reshape(1, -1))
    prediction = model.predict(future_features)
    future_predictions.append(prediction[0])
    
    # 다음 예측을 위해 미래 데이터 갱신
    if i + 1 < 48:
        future_data.at[future_dates[i + 1], 'lag1'] = prediction[0]
        if i + 4 < 48:
            future_data.at[future_dates[i + 4], 'lag4'] = prediction[0]
        
        rolling_window = future_predictions[-4:] if len(future_predictions) >= 4 else [last_values['price_log_diff']] * (4 - len(future_predictions)) + future_predictions
        future_data.at[future_dates[i + 1], 'rolling_mean_4'] = np.mean(rolling_window)
        future_data.at[future_dates[i + 1], 'rolling_std_4'] = np.std(rolling_window)

# 로그 역변환
future_pred_cumsum = pd.Series(future_predictions, index=future_dates).cumsum()
last_log_price = data['price_log'].iloc[-1]

future_pred_reversed = np.exp(future_pred_cumsum + last_log_price)

# 미래 예측 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(data.index, np.exp(data['price_log']), label='Historical Prices')
plt.plot(future_pred_reversed.index, future_pred_reversed, label='Future Predicted Prices', color='red')
plt.title('Future Rebar Prices Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# 미래 예측 결과 출력
print(future_pred_reversed.values)

df_test['철근 고장력 HD10mm 현물KRW/ton'] = future_pred_reversed.values

df_test

df_test.to_csv("sample_submission.csv", index=False)
