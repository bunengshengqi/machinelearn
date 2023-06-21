"""
此部分接chater10中的descision_tree_demo.py
"""
# 实例化一个随机森林
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
# 通过超参数调优
from sklearn.model_selection import GridSearchCV
param = {"n_estimators":[100, 120, 300], "max_depth": [3, 7, 11]}
gc = GridSearchCV(rf, param_grid=param, cv=3)
gc.fit(x_train, y_train)
print("随机森林预测精度是：\n",gc.score(x_test, y_test))