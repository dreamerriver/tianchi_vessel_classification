from train import train
from predict import  predict_data
from train import train_data
import numpy as np
import pandas as pd
#from lightgbm import LGBMClassifier
#est_bin是做了每个特征dbscan分箱
#train_x,train_y,est_kbin = train_data()
train_x,train_y = train_data()
train_x.drop(['lat_var_skew', 'lon_var_skew', 'velocity_var_skew',
       'velocity_var_covar', 'angle_var_mean', 'computed_angle_var_mean',
       'computed_angle_var_skew', 'computed_angle_var_covar',
       #'angle_var_velocity_cov'
       ],axis=1,inplace=True)

#构建测试集数据
#test_x,test_id = predict_data(est_kbin)
test_x,test_id = predict_data()
test_x.drop(['lat_var_skew', 'lon_var_skew', 'velocity_var_skew',
       'velocity_var_covar', 'angle_var_mean', 'computed_angle_var_mean',
       'computed_angle_var_skew', 'computed_angle_var_covar',
       #'angle_var_velocity_cov' 
       ],axis=1,inplace=True)

train_id = train_x.pop('id')

#开始训练 
clf = train(train_x,train_y)
print("训练完成…………")
#开始预测



# 对未知数据进行训练
y_test_proba = clf.predict_proba(test_x)  # 概率预测
#classes = np.unique()
y_test_prob = pd.concat([test_id,pd.DataFrame(y_test_proba)],axis=1)

y_test = y_test_prob.groupby('id').mean()

result = [clf.classes_[i] for i in np.argmax(y_test.values,axis=1)]
#result = result.map(lambda x: classes_[x])
result = pd.concat([pd.Series(y_test.index),pd.Series(result)],axis=1)

result.set_index('id',inplace=True)

result.to_csv('/result.csv',header=None,encoding='utf-8')