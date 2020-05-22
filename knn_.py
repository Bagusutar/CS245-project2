from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import numpy as np

dim = 2048
features = np.load("data/feature_{}.npy".format(dim))
labels = np.load("data/label.npy")

# 把要调整的参数以及其候选值 列出来；
param_grid = {"n_neighbors": range(1, 31)}
metric = "cosine"
knn = KNeighborsClassifier(metric=metric, algorithm="brute", weights="distance")

grid_search = GridSearchCV(knn, param_grid, cv=5, verbose=10, n_jobs=5)  # 实例化一个GridSearchCV类
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.4, random_state=10,
                                                    stratify=labels)
grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
test_score = grid_search.score(X_test, y_test)

print("Parameters:{}".format(param_grid))
print("Best parameters:{}".format(grid_search.best_params_))
print("Best score on train set:{:.4f}".format(grid_search.best_score_))
print("Test set score:{:.4f}".format(grid_search.score(X_test, y_test)))

results = grid_search.cv_results_
results['best_params'] = grid_search.best_params_
results['best_score'] = grid_search.best_score_
results['test_score'] = test_score
np.save('result/{} {}.npy'.format(metric, dim), results)

'''
The format of cv_results_:
 |          {
 |          'param_kernel': masked_array(data = ['poly', 'poly', 'rbf', 'rbf'],
 |                                       mask = [False False False False]...)
 |          'param_gamma': masked_array(data = [-- -- 0.1 0.2],
 |                                      mask = [ True  True False False]...),
 |          'param_degree': masked_array(data = [2.0 3.0 -- --],
 |                                       mask = [False False  True  True]...),
 |          'split0_test_score'  : [0.80, 0.70, 0.80, 0.93],
 |          'split1_test_score'  : [0.82, 0.50, 0.70, 0.78],
 |          'mean_test_score'    : [0.81, 0.60, 0.75, 0.85],
 |          'std_test_score'     : [0.01, 0.10, 0.05, 0.08],
 |          'rank_test_score'    : [2, 4, 3, 1],
 |          'split0_train_score' : [0.80, 0.92, 0.70, 0.93],
 |          'split1_train_score' : [0.82, 0.55, 0.70, 0.87],
 |          'mean_train_score'   : [0.81, 0.74, 0.70, 0.90],
 |          'std_train_score'    : [0.01, 0.19, 0.00, 0.03],
 |          'mean_fit_time'      : [0.73, 0.63, 0.43, 0.49],
 |          'std_fit_time'       : [0.01, 0.02, 0.01, 0.01],
 |          'mean_score_time'    : [0.01, 0.06, 0.04, 0.04],
 |          'std_score_time'     : [0.00, 0.00, 0.00, 0.01],
 |          'params'             : [{'kernel': 'poly', 'degree': 2}, ...],
 |          }
'''
