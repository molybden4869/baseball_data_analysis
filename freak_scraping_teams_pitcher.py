#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author:  Daisuke Sugizaki
"""

import pandas as pd


data_all = []
data_all_ce = []
data_all_pa = []

# urlの取得
urls = []
# 2009年から2020年まで　
for year in range(9, 23, 1):
    # https://baseball-data.com/19/team/pitcher.html
    urls.append('https://baseball-data.com/' + "{0:02d}".format(year) + '/team/pitcher.html')

# 2021年のデータ
urls.append('https://baseball-data.com/team/pitcher.html')


# urlからデータを取得
year = 9
for url in urls:
    print(url)
    data = pd.io.html.read_html(url)
    data_ce = data[0]
    data_pa = data[1]
    if year == 9:
        data_ce.loc[:, 'チーム'] = '200' + str(year) + '_' + data_ce.loc[:, 'チーム']
        data_pa.loc[:, 'チーム'] = '200' + str(year) + '_' + data_pa.loc[:, 'チーム']
        # data_ce.to_csv('freak_data_team_pitcher_ce_' + '200' + str(year) + '.csv') # セリーグ
        # data_pa.to_csv('freak_data_team_pitcher_pa_' + '200' + str(year) + '.csv') # パリーグ
    else:
        data_ce.loc[:, 'チーム'] = '20' + str(year) + '_' + data_ce.loc[:, 'チーム']
        data_pa.loc[:, 'チーム'] = '20' + str(year) + '_' + data_pa.loc[:, 'チーム']
        # data_ce.to_csv('freak_data_team_pitcher_ce_' + '20' + str(year) + '.csv') # セリーグ
        # data_pa.to_csv('freak_data_team_pitcher_pa_' + '20' + str(year) + '.csv') # パリーグ
    year = year + 1
    data_all_ce.append(data_ce)
    data_all_pa.append(data_pa)

# 取得したデータの結合
data_ce_all = pd.concat(data_all_ce, ignore_index=True) # セリーグ
data_pa_all = pd.concat(data_all_pa, ignore_index=True) # パリーグ
data_all = pd.concat([data_ce_all, data_pa_all], ignore_index=True) # セパ混合
# 保存
data_ce_all.to_csv('freak_data_team_pitcher_ce_all.csv')
data_pa_all.to_csv('freak_data_team_pitcher_pa_all.csv')
data_all.to_csv('freak_data_team_pitcher_all.csv')

