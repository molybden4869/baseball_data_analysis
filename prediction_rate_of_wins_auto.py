#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:23:30 2022

@author: ds4869
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.figure as figure
from sklearn import metrics, svm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, ElasticNetCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.ensemble import RandomForestRegressor


 # PLS
max_component_number = 20
pls_components = np.arange(1, max_component_number+1)
 # Ridge
alphas_ridge = 2 ** np.arange(-5, 10, dtype = float)
 # Lasso
alphas_lasso = np.arange(0.01, 0.71, 0.01, dtype = float)
 # EN
alphas_elastic_net = np.arange(0.01, 1.00, 0.01, dtype = float)
l1_ratios_elastic_net = np.arange(0.01, 0.71, 0.01, dtype = float)
 # SVR(linear)
cs_svr_linear = 2 ** np.arange(-5, 5, dtype = float)
epsilons_svr_linear = 2 ** np.arange(-10, 0, dtype = float)
 # SVR(rbf)
cs_svr_rbf = 2 ** np.arange(-5, 10, dtype = float)
epsilons_svr_rbf = 2 ** np.arange(-10, 0, dtype = float)
gammas_svr_rbf = 2 ** np.arange(-20, 10, dtype = float)
 # RF
rf_number_of_trees = 300 
rf_x_variables_rates = np.arange(0.1, 1.0, 0.1, dtype = float)


components = []
variance_of_gram_matrix = []
rmse_oob_all = []


# データの前処理方法の選択
 # 1:変数選択せずにそのまま使用, 2:説明変数間で相関の高い2つの変数のうち一方を削除
preprocessing_method = 1
 # 相関係数の閾値の設定
threshold_of_r = 0.97

deleting_variable_numbers_in_r = []


# 回帰分析手法のリスト
regression_list = ['OLS', 'PLS', 'RR', 'LASSO', 'EN', 'LSVR', 'NLSVR', 'RF', 'GPR']


# 回帰分析手法ごとの結果を格納
evaluation = ['r2_train', 'RMSE_train', 'MAE_train', 'r2_test', 'RMSE_test', 'MAE_test']
evaluation_all = pd.DataFrame(np.zeros([len(regression_list), len(evaluation)]))


 # 図を保存するディテクトリの作成
path_name = '結果（図）/'
os.makedirs(path_name, exist_ok=True)


# データの読み込み
 # training data
train_data = pd.read_csv('prediction_rate_of_wins_train.csv', index_col=0, encoding='shift_jis')
 # test data
test_data = pd.read_csv('prediction_rate_of_wins_test.csv', index_col=0, encoding='shift_jis')


# データの分割（train, y_test）
x_train = train_data.iloc[:, 5:39]
y_train = train_data.iloc[:, 39]
x_test = test_data.iloc[:, 5:39]
y_test = test_data.iloc[:, 39]

# cvの各パラメータ
fold_number = int(len(x_train) / 12)

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



# 9つの回帰分析手法で cvの実行と最良なハイパーパラメータでのモデル構築
for regression_method in range(len(regression_list)):
    print('回帰分析手法の番号：', regression_method ,'/', len(regression_list)-1)
    
    # r2を格納するリスト
    r2_cv_all = []
    
    
    if regression_method==0: # OLS
        model = LinearRegression()
    
    
    elif regression_method==1: # PLS
        for component in pls_components:
            model_cv = PLSRegression(n_components=component)
            estimated_y_cv = pd.DataFrame(model_selection.cross_val_predict(
                model_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
            estimated_y_cv = estimated_y_cv * y_train.std() + y_train.mean()
            r2_cv = metrics.r2_score(y_train, estimated_y_cv)
            r2_cv_all.append(r2_cv)
        optimal_component_number = pls_components[r2_cv_all.index(max(r2_cv_all))]
        print(optimal_component_number)
        model = PLSRegression(n_components=optimal_component_number)
        

    elif regression_method==2: # Ridge
        for alpha_ridge in alphas_ridge:
            model_cv = Ridge(alpha=alpha_ridge)
            estimated_y_cv = pd.DataFrame(model_selection.cross_val_predict(
                model_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
            estimated_y_cv = estimated_y_cv * y_train.std() + y_train.mean()
            r2_cv =  metrics.r2_score(y_train, estimated_y_cv)
            r2_cv_all.append(r2_cv)
        optimal_alpha_ridge = alphas_ridge[r2_cv_all.index(max(r2_cv_all))]
        print(optimal_alpha_ridge)
        model = Ridge(alpha=optimal_alpha_ridge)
        
            
    elif regression_method==3: # Lasso
        for alpha_lasso in alphas_lasso:
            model_cv = Lasso(alpha=alpha_lasso, tol=0.001)
            estimated_y_cv = pd.DataFrame(model_selection.cross_val_predict(
                model_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
            estimated_y_cv = estimated_y_cv * y_train.std() + y_train.mean()
            r2_cv =  metrics.r2_score(y_train, estimated_y_cv)
            r2_cv_all.append(r2_cv)
        optimal_alpha_lasso = alphas_lasso[r2_cv_all.index(max(r2_cv_all))]
        print(optimal_alpha_lasso)
        model = Lasso(alpha=optimal_alpha_lasso)
    
            
    elif regression_method==4: # EN
        model_cv = ElasticNetCV(alphas=alphas_elastic_net, l1_ratio=l1_ratios_elastic_net,
                                cv=fold_number)
        model_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_alpha_en = model_cv.alpha_
        optimal_l1_ratio_en = model_cv.l1_ratio_
        model = ElasticNet(alpha=optimal_alpha_en, l1_ratio=optimal_l1_ratio_en)
    

    elif regression_method==5: # SVR(linear)
        model_cv = GridSearchCV(svm.SVR(kernel = 'linear'), {'C': cs_svr_linear, 'epsilon': epsilons_svr_linear},
                                cv = fold_number)
        model_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_c_svr_linear = model_cv.best_params_['C']
        optimal_epsilon_svr_linear = model_cv.best_params_['epsilon']
        model = svm.SVR(kernel='linear', C=optimal_c_svr_linear, epsilon=optimal_epsilon_svr_linear)
    

    elif regression_method==6: # SVR(rbf)
        # グラム行列の分散が最大になるときの γ 
        numpy_autoscaled_x_train = np.array(autoscaled_x_train)
        for gamma_svr_rbf in gammas_svr_rbf:
            gram_matrix = np.exp(
                -gamma_svr_rbf * ((numpy_autoscaled_x_train[:, np.newaxis] - numpy_autoscaled_x_train) ** 2).sum(axis = 2))
            variance_of_gram_matrix.append(gram_matrix.var(ddof=1))
        optimal_gamma_svr_rbf = gammas_svr_rbf[
            np.where(variance_of_gram_matrix == np.max(variance_of_gram_matrix))[0][0]]
        variance_of_gram_matrix_copy = variance_of_gram_matrix.copy()
        variance_of_gram_matrix.clear()
        print(optimal_gamma_svr_rbf)
        # C, epsilon の最適化
        model_cv = GridSearchCV(svm.SVR(kernel='rbf', gamma=optimal_gamma_svr_rbf),
                                {'C': cs_svr_rbf, 'epsilon': epsilons_svr_rbf},
                                cv=fold_number)
        model_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_c_svr_rbf = model_cv.best_params_['C']
        optimal_epsilon_svr_rbf = model_cv.best_params_['epsilon']
        model = svm.SVR(kernel='rbf', gamma=optimal_gamma_svr_rbf,
                        C=optimal_c_svr_rbf, epsilon=optimal_epsilon_svr_rbf)
    

    elif regression_method==7: # RF
        # max_features
        max_features = np.round(x_train_new.shape[1] * rf_x_variables_rates)
        max_features_int = [int(s) for s in max_features]

        model_cv = GridSearchCV(RandomForestRegressor(n_estimators=rf_number_of_trees),
                                {'max_features': max_features_int}, cv=fold_number)
        model_cv.fit(autoscaled_x_train, autoscaled_y_train)
        optimal_max_features = model_cv.best_params_['max_features']
        model = RandomForestRegressor(n_estimators=rf_number_of_trees, max_features=optimal_max_features)
        
    
    elif regression_method==8: # GPR
        # kernels
        kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel(),
                   ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
                   ConstantKernel() * RBF(np.ones(x_train_new.shape[1])) + WhiteKernel(),
                   ConstantKernel() * RBF(np.ones(x_train_new.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
                   ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
                   ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                   ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
                   ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
                   ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
                   ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]
        
        for gp_kernel in kernels:
            model_cv = GaussianProcessRegressor(alpha=0, kernel=gp_kernel)
            estimated_y_cv = pd.DataFrame(model_selection.cross_val_predict(
                model_cv, autoscaled_x_train, autoscaled_y_train, cv=fold_number))
            estimated_y_cv = estimated_y_cv * y_train.std() + y_train.mean()
            r2_cv =  metrics.r2_score(y_train, estimated_y_cv)
            r2_cv_all.append(r2_cv)
        optimal_kernel = kernels[r2_cv_all.index(max(r2_cv_all))]
        print(optimal_kernel)
        model = GaussianProcessRegressor(kernel=optimal_kernel)
        

    model.fit(autoscaled_x_train, autoscaled_y_train)


    # trainデータの予測
    estimated_y_train = pd.DataFrame(model.predict(autoscaled_x_train), index=x_train.index,
                                     columns=['estimated_y_train'])
    estimated_y_train = estimated_y_train * y_train.std() + y_train.mean()
    
    # r2, RMSE, MAEの計算と保存
    r2_train = metrics.r2_score(y_train, estimated_y_train)
    evaluation_all.iloc[regression_method, 0] = r2_train   # r2の格納
    RMSE_train = metrics.mean_squared_error(y_train, estimated_y_train, squared=False)
    evaluation_all.iloc[regression_method, 1] = RMSE_train   # RMSEの格納
    MAE_train = metrics.mean_absolute_error(y_train, estimated_y_train)
    evaluation_all.iloc[regression_method, 2] = MAE_train   # MAEの格納

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
    plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
    plt.tight_layout()
    
    # 画像の保存
    filename = path_name + '{0}_train'.format(regression_list[regression_method])
    plt.savefig(filename, dpi=300) 
    
    plt.show() 


    # testデータの予測
    estimated_y_test = pd.DataFrame(model.predict(autoscaled_x_test), index=x_test.index,
                                    columns=['estimated_y_test'])
    estimated_y_test = estimated_y_test * y_train.std() + y_train.mean()
    
     # r2, RMSE, MAEの計算と保存
    r2_test = metrics.r2_score(y_test, estimated_y_test)
    evaluation_all.iloc[regression_method, 3] = r2_test  # r2の格納
    RMSE_test = metrics.mean_squared_error(y_test, estimated_y_test, squared=False)
    evaluation_all.iloc[regression_method, 4] = RMSE_test  # RMSEの格納
    MAE_test = metrics.mean_absolute_error(y_test, estimated_y_test)
    evaluation_all.iloc[regression_method, 5] = MAE_test  # MAEの格納

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
    plt.xlabel('2022年の実際の勝率', fontname="MS Gothic")
    plt.ylabel('2022年の予測された勝率', fontname="MS Gothic")
    plt.gca().set_aspect('equal', adjustable='box')  # 図の形を正方形に
    plt.tight_layout()
    
    # 画像の保存
    filename = path_name + '{0}_test'.format(regression_list[regression_method])
    plt.savefig(filename, dpi=300) 
    
    plt.show() 


# 結果の保存
evaluation_all.index = regression_list
evaluation_all.columns = evaluation
evaluation_all.to_csv('evaluation_all.csv', encoding='shift_jis')



    