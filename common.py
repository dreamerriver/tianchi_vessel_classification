import pandas as pd
import numpy as np
import multiprocessing 
from datetime import datetime
import os
import math
#from sklearn.preprocessing import KBinsDiscretizer
#import itertools
#import nolds
#汉明距离
def haversine_array(lat1, lng1, lat2, lng2):
     lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
     AVG_EARTH_RADIUS = 6371 # in km
     lat = lat2 - lat1
     lng = lng2 - lng1
     d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
     h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
     return h
#隐曼哈顿距离
def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
     a = haversine_array(lat1, lng1, lat1, lng2)
     b = haversine_array(lat1, lng1, lat2, lng1)
     return a + b
#
def bearing_array(lat1, lng1, lat2, lng2):
     lng_delta_rad = np.radians(lng2 - lng1)
     lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
     y = np.sin(lng_delta_rad) * np.cos(lat2)
     x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
     return np.degrees(np.arctan2(y, x))    
    
#构造统计特征
#返回值是个字典
def stat_feature(data_):
    result_ = {}
    for feature in data_.columns:
      if(feature in {'渔船ID','time','type'}):
        continue
      data_min = data_[feature].min()
      data_max = data_[feature].max()
      data_mean = data_[feature].mean()
      data_std = data_[feature].std()
      data_skew = data_[feature].skew()
      data_kurt = data_[feature].kurt()
      #均方根？？
      data_rms=math.sqrt(pow(data_mean,2) + pow(data_std,2))
      #波形因子
      data_boxing=data_rms / ((abs(data_[feature]).mean())+0.0001)
      #峰值因子
  
      data_fengzhi=(max(data_[feature])) / (data_rms+0.00001)
      #脉冲因子
      data_maichong=(max(data_[feature])) / ((abs(data_[feature]).mean())+0.00001)
      #裕度因子
      data_yudu=(max(data_[feature])) / (pow((data_[feature].map(lambda x: math.sqrt(abs(x))).sum()/(len(data_))),2)+0.00001)
      data_range = data_max - data_min
      data_covar = data_std/(data_mean+0.0001)

      result__ = {
              feature+'_min': data_min,
              feature+'_max': data_max,
              feature+'_mean':data_mean,
              feature+'_std':data_std,
              feature+'_skew':data_skew,
              feature+'_kurt':data_kurt,
              feature+'_range':data_range,
              feature+'_covar':data_covar,
              feature+'_rms':data_rms,
              feature+'_boxing':data_boxing,
              feature+'_fengzhi':data_fengzhi,
              feature+'_maichong':data_maichong,
              feature+'_yudu':data_yudu,

                  }
      result_.update(result__)

    return result_
def construct_newcol(data):
  data['lat_var'] = data['lat'].diff()
  #坐标y的变化
  data['lon_var'] = data['lon'].diff()
  #距离的变化
  data['distance_var'] = np.sqrt(data['lat_var'].values*data['lat_var'].values + data['lon_var'].values*data['lon_var'].values)
  #速度的变化
  data['velocity_var'] = data['速度'].diff()
  #角度的变化
  data['angle_var'] = data['方向'].diff()
  #计算出来的角度
  data['computed_angle'] = np.arctan(data['lon_var'].values/(data['lat_var'].values+0.00001))
  #计算坐标计算出来的角速度
  data['computed_angle_var'] = data['computed_angle'].diff() 
  data['haversine'] = haversine_array(data['lat'].shift(-1).values,
                                     data['lon'].shift(-1).values,
                                     data['lat'].values,data['lon'].values)
  data['dummy_manhattan_distance'] = dummy_manhattan_distance(data['lat'].shift(-1).values,
                                                              data['lon'].shift(-1).values,
                                                              data['lat'].values,
                                                              data['lon'].values)
  data['bearing'] = bearing_array(data['lat'].shift(-1).values,
                                  data['lon'].shift(-1).values,
                                  data['lat'].values,data['lon'].values)
  data.dropna(inplace=True)
  return data
#def preprocessing(data,id,est_kbin):
def preprocessing(data,id):
  #数据构造新列
  data = construct_newcol(data)
  result = stat_feature(data)
  for i in [0.25,0.75]:
    result['lat_quantile_'+str(i)] = data['lat'].quantile(i)
    result['lon_quantile_'+str(i)] = data['lon'].quantile(i)
    result['速度_quantile_'+str(i)] = data['速度'].quantile(i)
    result['distance_var_quantile_'+str(i)] = data['distance_var'].quantile(i)
    result['velocity_var_quantile_'+str(i)] = data['velocity_var'].quantile(i)
    result['angle_var_quantile_'+str(i)] = data['angle_var'].quantile(i)
    
  
  
  result['latlon_cov'] = data['lat'].cov(data['lon'])
  #数据坐标信息错误的话 这个值应该会不一样
  result['distance_var_velocity_cov'] = data['distance_var'].cov(data['速度'])
  #数据坐标信息错误的话 这个值应该也会不一样
  result['angle_var_computed_angle_var_cov'] = data['angle_var'].cov(data['computed_angle_var'])
  #计算角速度速度的协方差
  #result['angle_var_velocity_cov'] = data['angle_var'].cov(data['速度'])
  #计算角速度，加速度协方差
  result['angle_var_velocity_var_cov'] = data['angle_var'].cov(data['velocity_var']) 
  #result['xy_cov'] = data['x'].cov(data['y'])
  result['max_area'] = (result['lat_max'] - result['lat_min'])*(result['lon_max']-result['lon_min'])
  result['circle_area'] = pow(result['lat_max'] - result['lat_min'] + result['lon_max']-result['lon_min'],2)
  result['line_len'] = result['lat_max'] - result['lat_min'] + result['lon_max']-result['lon_min']
  #result['xy_area'] = (result['x_max'] - result['x_min'])*(result['y_max']-result['y_min'])   
  #速度最多的值
  result['velocity_first_counts'] = pd.cut(data['速度'].values,np.linspace(0,50,51)).value_counts().index[0].left
  #速度第二多的值
  result['velocity_second_counts'] = pd.cut(data['速度'].values,np.linspace(0,50,51)).value_counts().index[1].left
  #速度第三多的值
  result['velocity_third_counts'] = pd.cut(data['速度'],np.linspace(0,50,51)).value_counts().index[2].left
  #构造直方图离散化计数特征
  result['velocity_count_rate_1'] = len(data[data['速度']<1])/len(data)
  result['velocity_count_rate_2'] = len(data[(data['速度']<2) & (data['速度']>1)])/len(data)
  result['velocity_count_rate_3'] = len(data[(data['速度']<5) & (data['速度']>2)])/len(data)
  result['velocity_count_rate_4'] = len(data[(data['速度']<7) & (data['速度']>5)])/len(data)
  result['velocity_count_rate_5'] = len(data[(data['速度']<10) & (data['速度']>7)])/len(data)
  result['velocity_count_rate_6'] = len( data[data['速度']>10])/len(data)  
  #当时的时间
  result['hour'] = np.array(data.index.hour + data.index.minute/60.0).mean()
  #result['hour_std'] = np.array(data.index.hour + data.index.minute/60.0).std()  
  result['id'] = id
  return result


'''
如果要改成多线程处理，逻辑上看的话
首先将需要处理的文件列出来
然后多线程处理每个文件
'''
def read_csv_train(filename):
        data_ = pd.read_csv(filename,index_col='time',parse_dates=True)
        data_.index = [datetime.strptime(i, '%m%d %H:%M:%S') for i in data_.index]
        data_type = data_['type'][0]
        #data_= data_.resample('5T').mean().interpolate()
        return data_,data_type


#读取数据集，记录不合并
def read_data(data_dir_pre):     
  data_dir = os.listdir(data_dir_pre)
  #进程数
  processnum = 16
  #读取文件
  data = {}
  proced_data = []
  count = 0
  data_type = {}
  '''
  读取文件单进程
  '''

  for filename in data_dir:
    count = count + 1
    if(count%10==0):
      print('正在读取第{}个文件'.format(count))
    data[filename.split('.',1)[0]] , data_type[filename.split('.',1)[0]] = read_csv_train(data_dir_pre + filename)
    #对数据进行插值
    data[filename.split('.',1)[0]] = data[filename.split('.',1)[0]].resample('5T').mean().interpolate()
    data[filename.split('.',1)[0]] = construct_newcol(data[filename.split('.',1)[0]])
    
  #将数据进行分箱统计
  merged_data = pd.concat(data,axis=0)
  merged_data = merged_data[merged_data.columns.drop(['渔船ID'])]
  #每个特征聚成三类
  #est_kbin = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
  #est_kbin.fit(merged_data)


  #预处理数据
  pool1 = multiprocessing.Pool(processes = processnum)
  data_async = []
  count = 0
  for i in data.keys():
    count = count+1
    #print('正在为数据的第{}个文件构造数据的特征'.format(count))
    if(count%10==0):
      print('正在为数据的第{}个文件构造数据的特征'.format(count))
    data_len = len(data[i])
    data_start = 0
    step = 50
    while(data_len>=step):
        #proc_ = pool1.apply_async(func =preprocessing,args = (data[i][data_start:(data_start+step)].copy(),i,est_kbin))
        proc_ = pool1.apply_async(func =preprocessing,args = (data[i][data_start:(data_start+step)].copy(),i))
        #data_async元素是元组，proc_为异步执行结果，data_type为y值
        data_async.append((proc_,data_type[i]))
        data_len = data_len -step
        data_start = data_start +step
  pool1.close()
  pool1.join()
  proced_data = [x[0].get() for x in data_async]
  data_type = [x[1] for x in data_async]
  for i in range(len(data_async)):
    proced_data[i]['type'] = data_type[i]
  #return proced_data,est_kbin
  return proced_data
def new_cols(data):
  result = data
  result['lat_boxing_lat_min_mul'] = data['lat_boxing'].values * data['lat_min'].values
  result['lon_boxing_lat_min_mul'] = data['lon_boxing'].values * data['lat_min'].values
  result['lat_quantile_0.25_lat_min_mul'] = data['lat_quantile_0.25'].values * data['lat_min'].values
  result['lat_boxing_lat_max_mul'] = data['lat_boxing'].values * data['lat_max'].values
  result['lat_fengzhi_lat_max_mul'] = data['lat_fengzhi'].values * data['lat_max'].values
  result['lon_boxing_lat_max_mul'] = data['lon_boxing'].values * data['lat_max'].values
  result['lon_min_lat_boxing_mul'] = data['lon_min'].values * data['lat_boxing'].values
  result['lon_max_lat_boxing_mul'] = data['lon_max'].values * data['lat_boxing'].values
  result['lon_boxing_lat_boxing_mul'] = data['lon_boxing'].values * data['lat_boxing'].values
  result['lon_max_lat_fengzhi_mul'] = data['lon_max'].values * data['lat_fengzhi'].values
  result['lon_boxing_lon_min_mul'] = data['lon_boxing'].values * data['lon_min']
  result['lon_quantile_0.25_lon_min_mul'] = data['lon_quantile_0.25'].values * data['lon_min'].values
  result['lon_boxing_lon_max_mul'] = data['lon_boxing'].values * data['lon_max'].values
  result['lon_quantile_0.25_lon_boxing_mul'] = data['lon_quantile_0.25'].values * data['lon_boxing'].values
  '''
  result['lon_min_div_lat_boxing'] = data['lon_min'].values/(data['lat_boxing'].values + 0.0000000001)
  result['lon_min_div_lon_boxing'] = data['lon_min'].values/(data['lon_boxing'].values + 0.0000000001)
  result['lon_max_div_lat_boxing'] = data['lon_max'].values/(data['lat_boxing'].values + 0.0000000001)
  result['lon_max_div_lon_boxing'] = data['lon_max'].values/(data['lon_boxing'].values + 0.0000000001)
  result['lon_boxing_div_lon_max'] = data['lon_boxing'].values/(data['lon_max'].values + 0.0000000001)
  result['lon_quantile_0.25_div_lat_boxing'] = data['lon_quantile_0.25'].values/(data['lat_boxing'].values + 0.0000000001)
  '''
  return result

def read_csv_test(filename):
        data_ = pd.read_csv(filename,index_col='time',parse_dates=True)
        data_.index = [datetime.strptime(i, '%m%d %H:%M:%S') for i in data_.index]
        #data_= data_.resample('5T').mean().interpolate()
        return data_

#读取数据集，记录不合并
#def read_test(data_dir_pre,est_kbin):
def read_test(data_dir_pre):
  data_dir = os.listdir(data_dir_pre)
  #进程数
  processnum = 16
  #读取文件
  data = {}
  proced_data = []
  count = 0
  '''
  读取文件单进程
  '''
 
  for filename in data_dir:
    if(count%10==0):
      print('正在读取第{}个文件'.format(count))
    data[filename.split('.',1)[0]]= read_csv_test(data_dir_pre + filename)
    #对数据进行插值操作
    data[filename.split('.',1)[0]] = data[filename.split('.',1)[0]].resample('5T').mean().interpolate()
    #数据划分完成，都放入data字典中了
    
    #构造新的列
    data[filename.split('.',1)[0]] = construct_newcol(data[filename.split('.',1)[0]])

  #预处理数据
  pool2 = multiprocessing.Pool(processes = processnum)
  data_async = []
  count = 0
  for i in data.keys():
    count = count+1
    print('正在为数据的第{}个文件构造数据的特征'.format(count))
    if(count%10==0):
      print('正在为数据的第{}个文件构造数据的特征'.format(count))
    data_len = len(data[i])
    data_start = 0
    step = 50
    if(data_len<step):
        #proc_ = pool2.apply_async(func =preprocessing,args = (data[i][data_start:(data_start+step)].copy(),i,est_kbin))
        proc_ = pool2.apply_async(func =preprocessing,args = (data[i][data_start:(data_start+step)].copy(),i))
        data_async.append(proc_)
        data_len = data_len -step
        data_start = data_start +step
    while(data_len>=step):
        #proc_ = pool2.apply_async(func = preprocessing,args = (data[i][data_start:(data_start+step)].copy(),i,est_kbin))
        proc_ = pool2.apply_async(func =preprocessing,args = (data[i][data_start:(data_start+step)].copy(),i))
        data_async.append(proc_)
        data_len = data_len -step
        data_start = data_start +step
  pool2.close()
  pool2.join()
  proced_data = [x.get() for x in data_async]
  return proced_data
