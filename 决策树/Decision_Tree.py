# -*- ecoding: utf-8 -*-
# @ModuleName: create_tree
# @Function: 
# @Author: Yuxuan Xi
# @Time: 2020/5/19 8:38

import pandas as pd
import numpy as np
from collections import defaultdict
from math import log
import warnings
from common import *

class Node(object):
    def __init__(self,leaf=False,target=None,feature_name='null',sample_num=None):
        self.leaf=leaf
        self.target=target
        self.feature_name=feature_name
        #self.feature_id=feature_id
        self.sample_num=sample_num
        self.tree={}
        self.result={
            'target':self.target,
            'feature_name':self.feature_name,
            'sample_num':self.sample_num,
            'tree':self.tree
        }

    def add_node(self,value,node):
        if node:
            self.tree[value]=node

    def turn_to_leaf(self,target):
        self.target=target
        self.feature_name=None
        self.feature_id=None
        self.tree=None

    def print_node(self):
        print("{target:"+self.target+",feature:"+self.feature_name+",sample number:"+str(self.sample_num))
        if self.tree:
            print(self.tree.keys().__str__())
            for node in self.tree.values():
                node.print_node()



class Decision_Tree(object):
    def __init__(self,feature_types=None,ID3='IGR',epsilon=0,max_depth=None,min_samples_leaf=1,min_samples_split=2):#array,feature_types:[]每个feature是类别还是数字类型，
        # names:特征的名字,id3:计算IG还是IGR

        self.feature_types=feature_types
        #self.task=task
        self.ID3=ID3
        self.epsilon=epsilon
        self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf
        self.min_samples_split=min_samples_split
        self.tree={}
        self.features={}

    def cal_ent(self,labels):#labels:series
        num=len(labels)
        pro=labels.value_counts().values/num
        return -sum(map(lambda x:x*log(x,2),pro))

    '''count=defaultdict(int)
        for label in labels:
            count[label]+=1
        pro=count.values()/num
        return -sum(map(lambda x:x*log(x,2)))'''

    def find_best_feature(self,dataset):
        feature_names=dataset.columns.values[:-1]
        length=len(dataset)
        best_feat=feature_names[0]
        prim_ent=self.cal_ent(dataset['target'])
        dct={}
        for feature in feature_names:
            feat_values=dataset[feature].unique()
            rem=0
            for feat_value in feat_values:
                sub_data_label=dataset[dataset[feature]==feat_value]['target']
                rem+=len(sub_data_label)/length*self.cal_ent(sub_data_label)
            feat_ent=self.cal_ent(dataset[feature])
            IG=prim_ent-rem
            IGR=IG/(feat_ent+1e-10)#防止分母为0？
            if len(feat_values)>=self.min_samples_split and min(dataset[feature].value_counts())>self.min_samples_leaf:#预剪枝，只将满足可分的子树的数量大于最小可分数量纳入考虑
                dct[feature]=(IG,IGR)

        if dct=={}:
            return None
        if self.ID3=='IG':
            best_feat=max(dct.items(), key=lambda x: x[1][0])[0]
            cretirio=dct[best_feat][0]
        else:
            best_feat = max(dct.items(), key=lambda x: x[1][1])[0]
            cretirio = dct[best_feat][1]
        return best_feat if cretirio>self.epsilon else None

    def train(self,dataset,depth):
        if len(dataset)==0:
            return None
        target=dataset['target']
        if len(target.unique())==1:
            return Node(leaf=True,target=target.values[0],sample_num=len(dataset))
        best_feat=self.find_best_feature(dataset)
        major_target=target.value_counts().index.values[np.argmax(target.value_counts().values)]#找出该dataset中target占大多数的
        if best_feat==None or (self.max_depth!=None and depth>=self.max_depth):#如果不满足划分条件或者超过最大深度，不再划分设为叶子节点
            return Node(leaf=True,target=major_target,sample_num=len(dataset))
        node=Node(target=major_target,feature_name=best_feat,sample_num=len(dataset))#非叶子节点

        feat_values=dataset[best_feat].unique()
        for feat_value in feat_values:
            sub_dataset=dataset[dataset[best_feat]==feat_value].drop([best_feat], axis=1)
            if len(sub_dataset):node.add_node(feat_value,self.train(sub_dataset,depth+1))

        return node
    @Timer
    def fit(self,dataset):
        '''feats=dataset.columns.values
        for i,feat in enumerate(feats):
            self.features[feat]=i'''
        self.tree=self.train(dataset,0)#根节点深度为0

    def predict(self,data,tree):
        node=tree
        while(node.leaf==False):
            feat_value=data[node.feature_name]
            if feat_value not in node.tree:#如果验证集中的值没在训练集中出现，则直接进入第一个分支
                node=list(node.tree.values())[0]
            else:node=node.tree[feat_value]
        return node.target

    def evaluation(self,test_data,tree):#输入为dataframe，返回错误率

        total=len(test_data)
        if total==0:
            warnings.warn("the data to be evaluated is null")
        correct=0
        for i in range(total):
            y_pre=self.predict(test_data.iloc[i],tree)
            correct+=y_pre==test_data.iloc[i]['target']
        correct_ratio=correct/total
        #print("the correct ratio is %f, the error ratio is %f"%(correct_ratio,1-correct_ratio))
        return 1-correct_ratio

    def create_post_pruning_tree(self,valid_data,node):
        if node.leaf==True or len(valid_data)==0:
            return node
        for value,subnode in node.tree.items():
            sub_data=valid_data[valid_data[node.feature_name]==value]
            node.tree[value]=self.create_post_pruning_tree(sub_data,subnode)

        pruning_node=Node(leaf=True,target=node.target,sample_num=node.sample_num)
        if self.evaluation(valid_data,pruning_node)<self.evaluation(valid_data,node):
            return pruning_node
        else:return node

    def post_pruning(self,valid_data):#valid_data:dataframe
        self.tree=self.create_post_pruning_tree(valid_data,self.tree)

    def print_tree(self):
        self.tree.print_node()

