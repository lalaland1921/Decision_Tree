# -*- ecoding: utf-8 -*-
# @ModuleName: data_preprocess
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/5/19 8:24

import pandas as pd
from Decision_Tree import Decision_Tree
import numpy as np
from AdaBoost import AdaBoost_Classifier
import os


def preprocess_data(data_name,features):
    path=os.path.join('./data',data_name+'.data')

    if data_name=='soybean-large'  or data_name=='breast-cancer':#将target放在最后一列
        df = pd.read_csv(path, header=None, index_col=False, names=['target']+features )
        df=df.loc[:,features + ['target']]
    else:
        df = pd.read_csv(path, header=None, index_col=False, names=features + ['target'])
    df=df.replace('?',np.nan)#'?'表示缺失值
    df.dropna(axis=1,how='all')
    df.fillna(method='bfill')
    df.fillna(method='ffill')
    return df

data_names=['breast-cancer','car','soybean-large']
data_features=[['age','menopause','tumor-size','inv-nodes','node-caps','deg-malig','breast','breast-quad'],
                 ['buying','maint','doors','persons','lug_boot','safety'],
                 ['date','plant-stand','precip','temp','hail','crop-hist','area-damaged','severity','seed-tmt','germination',
                  'plant-growth','leaves','leafspots-halo','leafspots-marg','leafspot-size','leaf-shread','leaf-malf','leaf-mild',
                  'stem','lodging','stem-cankers','canker-lesion','fruiting-bodies','external decay','mycelium','int-discolor','sclerotia',
                  'fruit-pods','fruit spots','seed','mold-growth','seed-discolor','seed-size','shriveling','roots']
                 ]

if __name__ == '__main__':
    error_rates=[]
    run_times=[]
    for i in range(3):
        run_time=[]
        err_rate=[]
        df=preprocess_data(data_names[i],data_features[i])
        df=df.sample(frac=1).reset_index(drop=True)#shuffle the data
        df_len=len(df)
        train_ratio=0.6
        valid_ratio=0.2
        df_train=df[:int(df_len*train_ratio)]
        df_valid=df[int(df_len*train_ratio):int(df_len*(train_ratio+valid_ratio))]
        df_test=df[int(df_len*(1-train_ratio)):]
        dt_clf=Decision_Tree(ID3='IG')
        run_t,_=dt_clf.fit(df_train)
        run_time.append(run_t)
        dt_clf.print_tree()
        err=dt_clf.evaluation(df_test,dt_clf.tree)
        err_rate.append(err)
        #print("error rate:",err_rate)
        dt_clf.post_pruning(df_valid)
        dt_clf.print_tree()
        err=dt_clf.evaluation(df_test,dt_clf.tree)
        err_rate.append(err)
        dt_clf = Decision_Tree(ID3='IGR')
        run_t,_=dt_clf.fit(df_train)
        run_time.append(run_t)
        dt_clf.print_tree()
        err = dt_clf.evaluation(df_test, dt_clf.tree)
        err_rate.append(err)
        # print("error rate:",err_rate)
        dt_clf.post_pruning(df_valid)
        dt_clf.print_tree()
        err = dt_clf.evaluation(df_test, dt_clf.tree)
        err_rate.append(err)
        #print("error rate:", err_rate)

        adabst=AdaBoost_Classifier()
        run_t,_=adabst.fit(df_train.values)
        run_time.append(run_t)
        err=adabst.evaluation(df_test.values)
        err_rate.append(err)
        error_rates.append(err_rate)
        run_times.append(run_time)

    result=pd.DataFrame(error_rates,columns=['IG','IG-POST_PRONING','IGR','IGR-POST_PRONING','AdaBoost'],index=data_names)
    print(result)
    times=pd.DataFrame(run_times,columns=['IG','IGR','AdaBoost'],index=data_names)
    print(times)

