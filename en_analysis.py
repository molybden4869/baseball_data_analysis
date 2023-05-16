#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  Daisuke Sugizaki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sklearn import metrics
from sklearn.linear_model import ElasticNet, ElasticNetCV


# EN
alphas_elastic_net = np.arange(0.01, 1.00, 0.01, dtype = float)
l1_ratios_elastic_net = np.arange(0.01, 0.71, 0.01, dtype = float)


# データの読み込み
 # training data
train_data = pd.read_csv('prediction_rate_of_wins_train.csv', index_col=0, encoding='shift_jis')
 # test data
test_data = pd.read_csv('prediction_rate_of_wins_test.csv', index_col=0, encoding='shift_jis')
 # predict_data
predict_data = pd.read_csv('predict_data.csv', index_col=0, encoding='shift_jis')


# 全データでモデル構築するため結合
modeling_data = pd.concat([train_data, test_data], axis=0)


# データの分割（x, y）
x_model = modeling_data.iloc[:, 5:39]
y_model = modeling_data.iloc[:, 39]

# predict_dataを143試合分に換算
x_predict = predict_data.iloc[:, 1:39]

conversion_list = ['得点', '安打', '本塁打', '盗塁', '犠打', '四球', '死球',
                   '三振', '併殺打', 'セlブ', 'ホlルド', '完投', '完封勝', '被安打',
                   '被本塁打', '与四球', '与死球', '奪三振', '失点', '自責点']
# 143試合に換算
for i in x_predict.columns:
    if i in conversion_list:
        x_predict.loc[:,i] = round((x_predict.loc[:,i] / x_predict.loc[:, '試合']) * 143)

x_predict_conv = x_predict.iloc[:, 4:39]



# cvの各パラメータ
fold_number = int(len(x_model) / 12)

# 標準化
autoscaled_x_model = ( x_model - x_model.mean() ) / x_model.std()
autoscaled_y_model = ( y_model - y_model.mean() ) / y_model.std()
autoscaled_x_predict = ( x_predict_conv - x_model.mean() ) / x_model.std()


# ENでモデル構築
model_cv = ElasticNetCV(alphas=alphas_elastic_net, l1_ratio=l1_ratios_elastic_net,
                        cv=fold_number)
model_cv.fit(autoscaled_x_model, autoscaled_y_model)
optimal_alpha_en = model_cv.alpha_
optimal_l1_ratio_en = model_cv.l1_ratio_
model = ElasticNet(alpha=optimal_alpha_en, l1_ratio=optimal_l1_ratio_en)


model.fit(autoscaled_x_model, autoscaled_y_model)


# trainデータの予測
estimated_y_model = pd.DataFrame(model.predict(autoscaled_x_model), index=x_model.index,
                                  columns=['estimated_y_model'])
estimated_y_model = estimated_y_model * y_model.std() + y_model.mean()
 

# r2, RMSE, MAEの計算と保存
r2_model = metrics.r2_score(y_model, estimated_y_model)
print('r2_model:', r2_model)
RMSE_model = metrics.mean_squared_error(y_model, estimated_y_model, squared=False)
print('RMSE_model:', RMSE_model)
MAE_model = metrics.mean_absolute_error(y_model, estimated_y_model)
print('MAE_model:', MAE_model)


# yの実測値と予測値のプロット (y_model)
plt.rcParams['font.size'] = 14
plt.figure(figsize = figure.figaspect(1))
plt.scatter(y_model, estimated_y_model.iloc[:, 0], c = 'blue')
y_max = max(y_model.max(), estimated_y_model.iloc[:, 0].max())
y_min = min(y_model.min(), estimated_y_model.iloc[:, 0].min())
plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
         [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.xlabel('実際の勝率', fontname="MS Gothic")
plt.ylabel('勝率の予測値', fontname="MS Gothic")
plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に

plt.tight_layout()
plt.show() 


# predict データの予測
estimated_y_predict = pd.DataFrame(model.predict(autoscaled_x_predict), index=autoscaled_x_predict.index,
                                  columns=['estimated_y_predict'])
estimated_y_predict = estimated_y_predict * y_model.std() + y_model.mean()

# 保存
estimated_y_predict.to_csv('estimated_y_predict.csv', encoding='shift_jis')




