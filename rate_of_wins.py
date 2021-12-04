#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 02:52:07 2021

@author: ds4869
"""


import pandas as pd


# データセットの読み込み
df_team_hitter = pd.read_csv('freak_data_team_hitter_all.csv', index_col=0)
df_team_pitcher = pd.read_csv('freak_data_team_pitcher_all.csv', index_col=0)

# データの削除
df_team_pitcher_new = df_team_pitcher.drop(['順位', 'チーム', '試合', '勝利', '敗北', '引分'], axis = 1)

# データを結合して保存
data= pd.concat([df_team_hitter, df_team_pitcher_new], axis=1)
data.to_csv('freak_data_teams.csv')


# 勝率の計算
rate_of_wins = data['勝利'] / ( data['試合'] - data['引分'] )
rate_of_wins = pd.DataFrame(rate_of_wins, columns=['勝率'])


# 勝率の追加
data_new = pd.concat([data, rate_of_wins], axis=1)
data_new.index = data_new.iloc[:, 1]
data_new = data_new.drop('チーム', axis=1)
data_new.to_csv('freak_data_teams_with_wins.csv')

# 相関係数の確認
df = data_new.iloc[:, 5:]
coef_rate_of_wins = df.corr().iloc[:, 34]


# モデル構築用のデータ作成
# 2020年のデータは削除（試合数が120試合と少ないため）
modeling_data = data_new.loc[data_new['試合']!=120, :]
# test データ(2021年のデータ)
test_data =  modeling_data[modeling_data.index.str.contains('2021')]
test_data.to_csv('prediction_rate_of_wins_test.csv')
# train　データ（2009〜2019年のデータ）
train_data = modeling_data.drop(test_data.index, axis=0)
train_data.to_csv('prediction_rate_of_wins_train.csv')

