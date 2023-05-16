#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  Daisuke Sugizaki
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sklearn import metrics


# データの読み込み
# training data
train_data = pd.read_csv('prediction_rate_of_wins_train.csv', index_col=0)
# test data
test_data = pd.read_csv('prediction_rate_of_wins_test.csv', index_col=0)


# 2022年のピタゴラス勝率の計算
runs_scored = test_data.loc[:, '得点']
runs_against = test_data.loc[:, '失点']


# 指数 2.0
pythagorean_expectation_2 = runs_scored**2 / (runs_scored**2 + runs_against**2)

# 実際の勝率
y = test_data.iloc[:, -1]


# r2, RMSE, MAEの計算と保存
r2_2 = metrics.r2_score(y, pythagorean_expectation_2)
print('r2(2.0):', r2_2)
RMSE_2 = metrics.mean_squared_error(y, pythagorean_expectation_2, squared=False)
print('RMSE(2.0):', RMSE_2)
MAE_2 = metrics.mean_absolute_error(y, pythagorean_expectation_2)
print('MAE(2.0):', MAE_2)


# yの実測値と予測値のプロット (y_train)
plt.rcParams['font.size'] = 12
plt.figure(figsize = figure.figaspect(1))
plt.scatter(y, pythagorean_expectation_2, c = 'blue')
y_max = max(y.max(), pythagorean_expectation_2.max())
y_min = min(y.min(), pythagorean_expectation_2.min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('実際の勝率', fontname="MS Gothic")
plt.ylabel('ピタゴラス勝率（指数 2.0)', fontname="MS Gothic")
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.tight_layout()
plt.show() 



# 指数 1.83
pythagorean_expectation_183 = runs_scored**1.83 / (runs_scored**1.83 + runs_against**1.83)


# r2, RMSE, MAEの計算と保存
r2_183 = metrics.r2_score(y, pythagorean_expectation_183)
print('r2(1.83):', r2_183)
RMSE_183 = metrics.mean_squared_error(y, pythagorean_expectation_183, squared=False)
print('RMSE(1.83):', RMSE_183)
MAE_183 = metrics.mean_absolute_error(y, pythagorean_expectation_183)
print('MAE(1.83):', MAE_183)


# yの実測値と予測値のプロット (y_train)
plt.rcParams['font.size'] = 12
plt.figure(figsize = figure.figaspect(1))
plt.scatter(y, pythagorean_expectation_183, c = 'blue')
y_max = max(y.max(), pythagorean_expectation_183.max())
y_min = min(y.min(), pythagorean_expectation_183.min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('実際の勝率', fontname="MS Gothic")
plt.ylabel('ピタゴラス勝率（指数 1.83)', fontname="MS Gothic")
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.tight_layout()
plt.show() 



# PythagenPat
runs_scored_all = test_data.loc[:, '得点'].sum()
runs_against_all = test_data.loc[:, '失点'].sum()
games = test_data.loc[:, '試合'].sum()


X = ( ( runs_scored_all + runs_against_all ) / games )**0.287

pythagenpat_expectation = runs_scored**X / (runs_scored**X + runs_against**X)


# r2, RMSE, MAEの計算と保存
r2 = metrics.r2_score(y, pythagenpat_expectation)
print('r2(pythagenpat):', r2)
RMSE = metrics.mean_squared_error(y, pythagenpat_expectation, squared=False)
print('RMSE(pythagenpat):', RMSE)
MAE= metrics.mean_absolute_error(y, pythagenpat_expectation)
print('MAE(pythagenpat):', MAE)


# yの実測値と予測値のプロット (y_train)
plt.rcParams['font.size'] = 12
plt.figure(figsize = figure.figaspect(1))
plt.scatter(y, pythagenpat_expectation, c = 'blue')
y_max = max(y.max(), pythagenpat_expectation.max())
y_min = min(y.min(), pythagenpat_expectation.min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('実際の勝率', fontname="MS Gothic")
plt.ylabel('PythagenPat', fontname="MS Gothic")
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
plt.tight_layout()
plt.show() 







