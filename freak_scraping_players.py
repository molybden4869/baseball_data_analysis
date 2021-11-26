#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 12:54:31 2021

@author: ds4869
"""

import pandas as pd


data_regular = []
data_all = []

# urlの取得
urls_1 = [] # 規定打席
urls_2 = [] # 全て
# 2009年から2020年まで
for year in range(9, 21, 1):
    # https://baseball-data.com/20/stats/hitter-all/tpa-3.html : 規定打席
    # https://baseball-data.com/20/stats/hitter-all/tpa-1.html : 全て
    urls_1.append('https://baseball-data.com/' + "{0:02d}".format(year) + '/stats/hitter-all/tpa-3.html')
    urls_2.append('https://baseball-data.com/' + "{0:02d}".format(year) + '/stats/hitter-all/tpa-1.html')


# urlからデータを取得
# 規定打席
year = 9
for url_1 in urls_1:
    print(url_1)
    data_1 = pd.io.html.read_html(url_1)
    data_1 = data_1[0]
    data_1.columns = data_1.columns.droplevel(0)
    if year == 9:
        data_1.loc[:, '選手名'] = '200' + str(year) + '_' + data_1.loc[:, '選手名']
    else:
        data_1.loc[:, '選手名'] = '20' + str(year) + '_' + data_1.loc[:, '選手名']
    year = year + 1 
    data_regular.append(data_1)

year = 9 # url_2用に再度設定
for url_2 in urls_2:
    print(url_2)
    data_2 = pd.io.html.read_html(url_2)
    data_2 = data_2[0]
    data_2.columns = data_2.columns.droplevel(0)
    if year == 9:
        data_2.loc[:, '選手名'] = '200' + str(year) + '_' + data_2.loc[:, '選手名']
        data_2.to_csv('freak_data_player_all_200' + str(year) + '.csv') # 年度別に保存
    else:
        data_2.loc[:, '選手名'] = '20' + str(year) + '_' +   data_2.loc[:, '選手名']
        data_2.to_csv('freak_data_player_all_20' + str(year) + '.csv') # 年度別に保存
    year = year + 1
    data_all.append(data_2)


# 取得したデータの結合
 # 規定打席
data_players_regular = pd.concat(data_regular, ignore_index=True) # セパ混合
data_players_regular = data_players_regular.drop('順位', axis=1)
 # 全データ
data_players_all = pd.concat(data_all, ignore_index=True) # セパ混合
data_players_all = data_players_all.drop('順位', axis=1)
# 保存
data_players_regular.to_csv('freak_data_players_regular.csv')
data_players_all.to_csv('freak_data_players_all.csv')




