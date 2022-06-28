import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


def fill_nan(dataframe):
    dataframe['precipitation'] = dataframe['precipitation'].fillna(0)
    # dataframe = dataframe.dropna()
    dataframe = dataframe.fillna(dataframe.mean())
    return dataframe


def seperate_datetime(dataframe):
    week_list = []
    year = []
    month = []
    for date in dataframe.date:
        year_point, month_point, day_point = date.split('-')
        year.append(int(year_point) - 2017)
        month.append(int(month_point))
    dataframe['year'] = year
    dataframe['month'] = month
    for day in dataframe['date']:
        num = pd.date_range(day, day, freq='D').to_series()
        week_list.append(int(num.dt.dayofweek[0]))
    dataframe['day'] = week_list
    dataframe = dataframe.drop(['date'], axis=1)
    return dataframe


def weekday_onehotcode(dataframe):
    new = pd.DataFrame()
    for i in range(0, 7):
        a = dataframe[dataframe['day'] == i]
        a['day'] = f'{i}'
        new = pd.concat([new, a], axis=0)
    new = pd.get_dummies(new)
    return new


def month_onehotcode(dataframe):
    new = pd.DataFrame()
    for i in range(1, 13):
        a = dataframe[dataframe['month'] == i]
        a['month'] = f'{i}'
        new = pd.concat([new, a], axis=0)
    new = pd.get_dummies(new)
    return new



def year_onehotcode(dataframe):
    a = dataframe[dataframe['year'] == 1]
    a['year'] = 'a'
    b = dataframe[dataframe['year'] == 2]
    b['year'] = 'b'
    c = dataframe[dataframe['year'] == 3]
    c['year'] = 'c'
    dataframe = pd.concat([a, b, c], axis=0)
    dataframe = pd.get_dummies(dataframe)
    return dataframe

def year_onehotcode_test(dataframe):
    c = dataframe[dataframe['year'] == 4]
    c['year'] = 'c'
    dataframe = pd.get_dummies(dataframe)
    dataframe['year_a'] = 0
    dataframe['year_b'] = 0
    return dataframe


def rental_rate(dataframe):
    y1 = dataframe[dataframe['year'] == 1]['rental'] * 2.3
    y2 = dataframe[dataframe['year'] == 2]['rental'] * 1.2
    y3 = dataframe[dataframe['year'] == 3]['rental']
    new = pd.concat([y1, y2, y3], axis=0).to_frame()
    dataframe['rental'] = new['rental']
    return True


def NMAE(true, pred):
    score = np.mean(np.abs(true - pred) / true)
    print(score)
    return score


def enter_week(dataframe):
    w_list = [0.993, 1.049, 0.998, 1.013, 1.049, 0.994, 0.903]
    for i in range(0, 7):
        dataframe[dataframe['day'] == i] *= w_list[i]


def enter_month(dataframe):
    m_list = [0.324, 0.342, 0.646, 0.992, 1.264, 1.442, 1.001, 1.084, 1.475, 1.502, 1.032, 0.574]
    for i in range(0, 12):
        dataframe[dataframe['month'] == i+1] *= m_list[i]

def enter_week_train(dataframe):
    w_list = [0.993, 1.049, 0.998, 1.013, 1.049, 0.994, 0.903]
    new = pd.DataFrame()
    for i in range(0, 7):
        a = pd.DataFrame()
        a['rental'] = dataframe[dataframe['day'] == i]['rental'] / w_list[i]
        new = pd.concat([new, a],axis=0)
    return new

def enter_month_train(dataframe):
    m_list = [0.324, 0.342, 0.646, 0.992, 1.264, 1.442, 1.001, 1.084, 1.475, 1.502, 1.032, 0.574]
    new = pd.DataFrame()
    for i in range(1, 13):
        a = pd.DataFrame()
        a['rental'] = dataframe[dataframe['month'] == i]['rental'] / m_list[i]
        new = pd.concat([new, a],axis=0)
    return new

clist = ['temp_lowest*month_8','PM2.5*day_6', 'humidity*month_12']

bicycle = pd.read_csv('train.csv')
bicycle = seperate_datetime(bicycle)
bicycle = fill_nan(bicycle)
rental_rate(bicycle)
# rental = enter_week_train(bicycle)
# bicycle['rental'] = rental
# print(bicycle)
bicycle = weekday_onehotcode(bicycle)
bicycle = month_onehotcode(bicycle)
bicycle["rental"] = np.log1p(bicycle["rental"])


y = bicycle['rental']
bicycle = bicycle.drop(['rental'], axis=1)

scaler1 = QuantileTransformer()
scaler2 = QuantileTransformer()
col___list = ['PM10','PM2.5','sunshine_rate','sunshine_sum','wind_max']
# col___list = bicycle.columns
scaler1.fit(bicycle[col___list])
X_train_scaled = scaler1.transform(bicycle[col___list])
bicycle[col___list] =X_train_scaled
T = bicycle['temp_mean']  # 섭씨
V = bicycle['wind_mean']  # 바람
R = bicycle['humidity']  # 습도
H = (bicycle['temp_mean'] * 1.8) + 32  # 화씨
feel_degree = 13.12 + 0.6215 * T - 11.37 * V**0.16 + 0.3965 * V**0.16
badfeel_degree = 9/5 * T - 0.55 * (1 - R) * (9/5 * T - 26) + 32
heat_degree = -42.379 + 2.05901523 * H + 10.14333127 * R - 0.22475541 * H * R - 6.83783 * 10**-3 * H**2 - 5.481717 * 10**-2 * R**2 + 1.22874 * 10**-3 * H**2 * R + 8.5282 * 10**-4 * H * R**2 - 1.99 * 10**-6 * H**2 * R**2
bicycle['dust'] = bicycle['PM10'] * bicycle['PM2.5']
bicycle['temp_ratio'] = abs(bicycle['temp_highest'] - bicycle['temp_lowest'])
bicycle['daytime'] = bicycle['sunshine_rate'] / bicycle['sunshine_sum']
bicycle['daytime'] = bicycle['daytime'].fillna(method='bfill')
bicycle['feel_degree'] = feel_degree
bicycle['badfeel_degree'] = badfeel_degree

col_list = list(bicycle.columns)
for i in range(len(col_list)):
    for j in range(i, len(col_list)):
        bicycle[f'{col_list[i]}*{col_list[j]}'] = bicycle[col_list[i]] * bicycle[col_list[j]]

bicycle = bicycle.drop(clist, axis=1)
x = bicycle


bicycle_test = pd.read_csv('test.csv')
bicycle_test_date = bicycle_test['date']
bicycle_test = seperate_datetime(bicycle_test)
bicycle_test = fill_nan(bicycle_test)  # 바로 뒤에 값으로 결측치를 채우므로 seperate 밑이 와야함
bicycle_test = weekday_onehotcode(bicycle_test)
bicycle_test = month_onehotcode(bicycle_test)
bicycle_test = bicycle_test.sort_index()


scaler2.fit(bicycle_test[col___list])
X_train_scaled = scaler2.transform(bicycle_test[col___list])
bicycle_test[col___list] =X_train_scaled
T = bicycle_test['temp_mean']  # 섭씨
V = bicycle_test['wind_mean']  # 바람
R = bicycle_test['humidity']  # 습도
H = (bicycle_test['temp_mean'] * 1.8) + 32  # 화씨
feel_degree = 13.12 + 0.6215 * T - 11.37 * V**0.16 + 0.3965 * V**0.16
badfeel_degree = 9/5 * T - 0.55 * (1 - R) * (9/5 * T - 26) + 32
heat_degree = -42.379 + 2.05901523 * H + 10.14333127 * R - 0.22475541 * H * R - 6.83783 * 10**-3 * H**2 - 5.481717 * 10**-2 * R**2 + 1.22874 * 10**-3 * H**2 * R + 8.5282 * 10**-4 * H * R**2 - 1.99 * 10**-6 * H**2 * R**2
bicycle_test['dust'] = bicycle_test['PM10'] * bicycle_test['PM2.5']
bicycle_test['temp_ratio'] = abs(bicycle_test['temp_highest'] - bicycle_test['temp_lowest'])
bicycle_test['daytime'] = bicycle_test['sunshine_rate'] / bicycle_test['sunshine_sum']
bicycle_test['feel_degree'] = feel_degree
bicycle_test['daytime'] = bicycle_test['daytime'].fillna(method='bfill')
bicycle_test['badfeel_degree'] = badfeel_degree

col_list = list(bicycle_test.columns)
for i in range(len(col_list)):
    for j in range(i, len(col_list)):
        bicycle_test[f'{col_list[i]}*{col_list[j]}'] = bicycle_test[col_list[i]] * bicycle_test[col_list[j]]

bicycle_test = bicycle_test.drop(clist, axis=1)
x2 = bicycle_test

X_train = x
y_train = y
X_test = x2

xg_reg = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, max_depth=5, n_estimators=1000)  # 나중에 늘리기 1000으로
xg_reg.fit(X_train, y_train)
pred = xg_reg.predict(X_test)
# xgb.plot_importance(xg_reg, max_num_features=10)
# plt.show()

feature_important = xg_reg.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
feature_dic = {}
for i in range(len(keys)):
    feature_dic[keys[i]] = values[i]
feature_list = sorted(feature_dic.items(),key=lambda x:x[1])
print(len(feature_list),feature_list[:150])

pred = pd.DataFrame(pred, columns=['rental'])
pred = np.expm1(pred) * 1.3
result = pd.concat([bicycle_test_date, pred],axis=1)
result = seperate_datetime(result)
enter_week(result)
# enter_month(result)
result = pd.concat([bicycle_test_date, result['rental']],axis=1)
real_data = pd.read_csv('real_data.csv')
NMAE(real_data['rental'],pred['rental'])
result.to_csv('sample_submissoin.csv',index=False)



