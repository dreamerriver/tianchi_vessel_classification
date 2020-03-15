import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.utils import class_weight

#尝试lightgbm
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  StackingClassifier
from sklearn.linear_model import LogisticRegressionCV
#from cleanlab.classification import LearningWithNoisyLabels

#from common import new_cols

from common import read_data



def train_data():
    train_ = read_data('./tcdata/hy_round2_train_20200225/')
    train_x = pd.DataFrame.from_dict(train_)
    train_y = train_x.pop('type')
    #测试得到结果需要id 
    #训练模型不需要
    #train_id = train_x.pop('id')   
    #train_x = new_cols(train_x)
    return train_x,train_y

def train(x,y):
    cw = list(class_weight.compute_class_weight('balanced',np.unique(y),y))
    lr = LogisticRegressionCV(cv=5,class_weight='balanced',scoring='f1_macro',verbose=10,random_state=0)  
    clf1 = lgb.LGBMClassifier(objective = 'multi:softmax',
                 n_estimators=900,#900
                 max_depth = 11,#8
                 num_leaves = 90,#90
                 learning_rate = 0.17,
                 feature_fraction = 0.7,
                 min_child_samples=5,
                 min_child_weight=0.001,
                 bagging_fraction = 1,
                 bagging_freq = 0,
                 reg_alpha = 0.015,
                 reg_lambda = 0,
                 cat_smooth = 0,
                 #device= 'gpu',
                 #gpu_platform_id= 1,
                 #gpu_device_id= 0,
                 class_weight='balanced',
                 random_state=0,
                 n_jobs=-1
                )
    #听说lightgbm里的 随机森林比sklearn要快？但是我莫名奇妙分数下去了，改回来了
    clf2 = RandomForestClassifier(n_estimators = 1000,random_state=0,n_jobs=-1,class_weight='balanced')
    clf3 = CatBoostClassifier(iterations=2000,verbose=400,early_stopping_rounds=200,#task_type='GPU',
                  #border_count=254,
            loss_function='MultiClass',
                 class_weights=cw,depth=8,l2_leaf_reg=0.06,random_strength=0.01,random_state=0)
    clf = StackingClassifier(estimators=[('lgb',clf1),('rf',clf2),('catboost',clf3)],cv=5,final_estimator=lr,
                 stack_method='predict_proba',verbose=10,n_jobs=1)
   
    clf.fit(x,y)
    return clf