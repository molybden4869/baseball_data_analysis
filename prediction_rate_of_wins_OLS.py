#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 17:33:59 2021

@author: ds4869
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sklearn import metrics
from sklearn.linear_model import LinearRegression


# データの読み込み
 # training data
train_data = pd.read_csv('prediction_rate_of_wins_train.csv', index_col=0)
 # test data
test_data = pd.read_csv('prediction_rate_of_wins_test.csv', index_col=0)


# データの前処理方法の選択
 # 1:変数選択せずにそのまま使用, 2:説明変数間で相関の高い2つの変数のうち一方を削除
preprocessing_method = 1
 # 相関係数の閾値の設定
threshold_of_r = 0.97
deleting_variable_numbers_in_r = []


# データの分割（train, y_test）
x_train = train_data.iloc[:, 5:39]
y_train = train_data.iloc[:, 39]
x_test = test_data.iloc[:, 5:39]
y_test = test_data.iloc[:, 39]


# データの前処理
if preprocessing_method == 1:
    x_train_new = x_train.copy()
    x_test_new = x_test.copy()

elif preprocessing_method == 2:
    
    # 相関行列を計算し絶対値をとる
    r_in_x = abs(x_train.corr())
    
    # 対角線（自分自身との相関を0にする）
    for i in range(r_in_x.shape[0]):
        r_in_x.iloc[i,i] = 0
    
    # 相関係数の絶対値が閾値以上となる変数の組み合わせのうち一方を削除
    for i in range(r_in_x.shape[0]):
        print(i+1, '/', r_in_x.shape[0])
        r_max = list(r_in_x.max())
        if max(r_max) >= threshold_of_r:
            variable_number_1 = r_max.index(max(r_max))
            r_in_variable_1 = list(r_in_x.iloc[:, variable_number_1])
            variable_number_2 = r_in_variable_1.index(max(r_in_variable_1))
            # 2つの変数における他の特徴量との相関係数の絶対値の和を計算
            sum_r_1 = r_in_x.iloc[:, variable_number_1].sum()
            sum_r_2 = r_in_x.iloc[:, variable_number_2].sum()
            if sum_r_1 > sum_r_2:
                delete_x_number = variable_number_1
            else:
                delete_x_number = variable_number_2
            deleting_variable_numbers_in_r.append(delete_x_number)
            # 削除する特徴量の影響をなくすため、対応する相関係数の絶対値を0にする
            r_in_x.iloc[:, delete_x_number] = 0
            r_in_x.iloc[delete_x_number, :] = 0
        else: # 相関係数の絶対値が閾値以上となる変数の組みがなくなったら終了
            break
    
    # 変数削除
    print(len(deleting_variable_numbers_in_r)) # 削除する変数の数の確認
    deleting_variables = list(x_train.columns[deleting_variable_numbers_in_r])
    x_train_new = x_train.drop(deleting_variables, axis=1)
    x_test_new = x_test.drop(deleting_variables, axis=1)


# 標準化
autoscaled_x_train = ( x_train_new - x_train_new.mean() ) / x_train_new.std()
autoscaled_y_train = ( y_train - y_train.mean() ) / y_train.std()
autoscaled_x_test = ( x_test_new - x_train_new.mean() ) / x_train_new.std()

# モデル構築
model = LinearRegression()
model.fit(autoscaled_x_train, autoscaled_y_train)

# 標準回帰係数の確認
standard_regression_coefficients = pd.DataFrame(model.coef_, index=x_train_new.columns,
                                                columns=['standard_regression_coefficients'])


# trainデータの予測
estimated_y_train = pd.DataFrame(model.predict(autoscaled_x_train), index=x_train.index,
                                 columns=['estimated_y_train'])
estimated_y_train = estimated_y_train * y_train.std() + y_train.mean()
estimated_y_train.to_csv('estimated_y_train.csv')

# yの実測値と予測値のプロット (y_train)
plt.rcParams['font.size'] = 18
plt.figure(figsize = figure.figaspect(1))
plt.scatter(y_train, estimated_y_train.iloc[:, 0], c = 'blue')
y_max = max(y_train.max(), estimated_y_train.iloc[:, 0].max())
y_min = min(y_train.min(), estimated_y_train.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('実際の勝率', fontname="MS Gothic")
plt.ylabel('勝率の予測値', fontname="MS Gothic")

# r2, RMSE, MAEの計算と保存
r2_train = metrics.r2_score(y_train, estimated_y_train)
print(r2_train)
RMSE_train = metrics.mean_squared_error(y_train, estimated_y_train, squared=False)
print(RMSE_train)
MAE_train = metrics.mean_absolute_error(y_train, estimated_y_train)
print(MAE_train)

evaluation_index_train = [r2_train, RMSE_train, MAE_train]
evaluation = pd.DataFrame(evaluation_index_train)
evaluation.index = ['r2', 'RMSE', 'MAE']
evaluation.columns = ['train']
evaluation.to_csv('evaluation_train.csv')


# testデータの予測
estimated_y_test = pd.DataFrame(model.predict(autoscaled_x_test), index=x_test.index,
                                columns=['estimated_y_test'])
estimated_y_test = estimated_y_test * y_train.std() + y_train.mean()
estimated_y_test.to_csv('estimated_y_test.csv')

# yの実測値と予測値のプロット (y_test)
plt.rcParams['font.size'] = 18
plt.figure(figsize = figure.figaspect(1))
plt.scatter(y_test, estimated_y_test.iloc[:, 0], c = 'blue')
y_max = max(y_test.max(), estimated_y_test.iloc[:, 0].max())
y_min = min(y_test.min(), estimated_y_test.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('2021年の実際の勝率', fontname="MS Gothic")
plt.ylabel('2021年の予測された勝率', fontname="MS Gothic")

# r2, RMSE, MAEの計算と保存
r2_test = metrics.r2_score(y_test, estimated_y_test)
print(r2_test)
RMSE_test = metrics.mean_squared_error(y_test, estimated_y_test, squared=False)
print(RMSE_test)
MAE_test = metrics.mean_absolute_error(y_test, estimated_y_test)
print(MAE_test)

evaluation_index_test = [r2_test, RMSE_test, MAE_test]
evaluation = pd.DataFrame(evaluation_index_test)
evaluation.index = ['r2', 'RMSE', 'MAE']
evaluation.columns = ['test']
evaluation.to_csv('evaluation_test.csv')

