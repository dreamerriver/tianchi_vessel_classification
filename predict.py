import pandas as pd
from common import read_test
#from common import new_cols

#def predict_data(est_kbin):
def predict_data():
    print('开始加载预测数据…………')
    #test_ = read_test('./tcdata/hy_round2_testA_20200225/',est_kbin)
    test_ = read_test('./tcdata/hy_round2_testB_20200312/')
    test_x = pd.DataFrame.from_dict(test_)
    #test_x = new_cols(test_x)
    test_id = test_x.pop('id')
    print('预测数据加载完成…………')
    return test_x,test_id