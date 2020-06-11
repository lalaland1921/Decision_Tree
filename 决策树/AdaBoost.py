# -*- ecoding: utf-8 -*-
# @ModuleName: AdaBoost
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/5/27 8:58

#from Decision_Tree import Decision_Tree
import numpy as np
from common import *

class AdaBoost_Classifier(object):#可进行多分类分类
    def __init__(self,n_estimators=50):
        self.n_estimators=n_estimators

    def weak_classifier(self,data:np.array,targets,w):#data:训练数据，w：权重,根据权重训练单层决策树
        tree={'error':1,'dim':0,'pos_values':[]}
        target_values=set(targets)
        m,n=data.shape
        for i in range(n):
            pos_values = []
            err=0
            for value in set(data[:,i]):
                sub_targets=targets[data[:,i]==value]
                weights=w[data[:,i]==value]
                if np.sum(weights[sub_targets==1])>np.sum(weights[sub_targets==-1]):
                    pos_values.append(value)
                    err+=np.sum(weights[sub_targets==-1])
            if err<tree['error']:
                tree['error']=err
                tree['dim']=i
                tree['pos_values']=set(pos_values)
        return tree

    def simple_predict(self,tree,data):
        m,n=data.shape
        targets=np.zeros(m)
        dim,pos_values=tree['dim'],tree['pos_values']
        for i in range(m):
            targets[i]=1 if data[i,dim] in pos_values else -1
        return targets

    def bin_fit(self,data,targets):#二分类
        m,n=data.shape
        w=np.ones(m)/m
        classifiers=[]
        for i in range(self.n_estimators):
            tree=self.weak_classifier(data,targets,w)
            pre_labels=self.simple_predict(tree,data)*targets#样本预测是否正确，正确为1，错误为-1
            err=tree['error']
            w_ch=1/(2*((pre_labels+1)//2-pre_labels*err)+1e-10)
            w*=w_ch
            clf={'dim':tree['dim'],'pos_values':tree['pos_values']}
            if err==0:#全都预测错误，把该分类器舍弃，实际是不可能发生的
                clf['alpha']=0
            else:clf['alpha']=1/2*np.log((1-err)/(1e-10+err))#
            classifiers.append(clf)
        return classifiers

    def bin_predict(self,data,bin_clf):#二分类预测
        '''if not self.classifiers:
            warnings.warn("the model is null, maybe you should train the model first")
            return'''
        m,n=data.shape
        pre=np.zeros(m)
        for tree in bin_clf:
            pre+=self.simple_predict(tree,data)*tree['alpha']
        return pre>0
    @Timer
    def fit(self,data):
        targets=data[:,-1]
        target_values=list(set(targets))

        self.classifiers={}
        subdata=data
        for i in range(len(target_values)-1):
            target=target_values[i]
            m,n=subdata.shape
            sub_targets=np.ones(m)
            sub_targets[subdata[:,-1]!=target]=-1
            self.classifiers[target]=self.bin_fit(subdata[:,:-1],sub_targets)
            subdata=subdata[subdata[:,-1]!=target]
        self.classifiers[target_values[-1]]=None

    def predict(self,data):
        m,_=data.shape
        targets=list(self.classifiers.keys())
        pre_num=np.zeros(m)-1
        index=np.ones(m)
        index=index.astype(bool)
        for i in range(len(targets)-1):
            bin_clf=self.classifiers[targets[i]]
            pre=self.bin_predict(data[index],bin_clf)
            tmp=pre_num[index]
            tmp[pre]=i
            pre_num[index]=tmp
            index = (pre_num==-1)#将剩下未分类的数据分出来
        pre_num[index]=len(targets)-1
        pre_num=pre_num.astype(int)
        total_pre=np.array([targets[i] for i in pre_num])
        return total_pre

    def evaluation(self,data):
        m,_=data.shape
        target=data[:,-1]
        pre=self.predict(data[:,:-1])
        error=np.sum(target!=pre)
        err_ratio=1.0*error/m
        print("the error rate is: ",err_ratio)
        return err_ratio

'''class AdaBoost_Classifier(object):#可实现多分类
    def __init__(self,n_estimators=50):
        self.n_estimators=n_estimators

    def build_simple_clf(self,data,w):#data:pd.dataframe
        features=data.columns[:-1]
        m=len(data)
        sim_clf={'min_err':1,'best_feat':features[0],'subtree':{}}
        target_values=data['target'].unique().values
        for feat in features:
            subtree={}
            err=0
            feat_values=data[feat].unique()
            for value in feat_values:
                sub_target=data[data[feat]==value]['target'].values
                sub_w=w[data[feat]==value]
                best_arg=np.argmax(np.array([sum(sub_w[sub_target==target]) for target in target_values]))
                subtree[value]=target_values[best_arg]
                err+=sum(sub_w[sub_target!=target_values[best_arg]])
            if err<sim_clf['min_err']:
                sim_clf['min_err']=err
                sim_clf['best_feat']=feat
                sim_clf['subtree']=subtree

        return sim_clf

    def sim_predict(self,sim_clf,data):
        pre=[]
        feat=sim_clf['best_feat']
        subtree=sim_clf['sub_tree']
        for i in range(len(data)):
            pre.append(subtree[data[feat][i]])
        return np.array(pre)

    def fit(self,data):'''
