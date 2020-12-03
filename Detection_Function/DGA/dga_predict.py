import pandas as pd 
import seaborn as sns
import numpy as np
import math
import collections
import os
import tldextract
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,feature_extraction
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import CuDNNLSTM,Conv1D,Input,concatenate
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras import models
from keras.layers.pooling import GlobalMaxPooling1D

class DGA_predictor(object):
    def __init__(self):
        self.malicous_dns_file='./Detection_Function/DGA/data/dga.txt'
        self.normal_dns_file='./Detection_Function/DGA/data/top-1m.csv'
        self.data_dir='./data/'
    
    def extract_domain_info(self,domain):
        ext=tldextract.extract(domain)
        subdomain,domain,suffix=ext.subdomain,ext.domain,ext.suffix
        return (subdomain,domain,suffix)

    def entropy_calculator(self,domain):
        counter_char= collections.Counter(domain)
        entropy=0
        for c,ctn in counter_char.items():
            _p = float(ctn)/len(domain)
            entropy+=-1*_p*math.log(_p,2)
        return round(entropy,7)

    def vowel_ratio(self,domain):
        vowel_list=['a','e','i','o','u']
        domain=domain.lower()
        vowel_count=0
        if len(domain)==0:
            return 0
        else:
            for i in range(0,len(domain)):
                if domain[i] in vowel_list:
                    vowel_count+=1
        vowel_ratio=vowel_count/len(domain)
        return round(vowel_ratio,7)

    def suffix_class(self,suffix):
        common_suffix=['cn','com','cc','net','org','gov','info']
        if suffix in common_suffix:return 0
        else:return 1

    def load_data(self):
        self.normal_dns_df=pd.read_csv(self.normal_dns_file,header=None,names=['No','Name'])
        self.normal_dns_df.drop(['No'],axis=1,inplace=True)
        self.malicous_dns_df=[]
        with open(self.malicous_dns_file,'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                temp_line=line.strip('\n').split('\t')
                if len(line)==1:
                    continue
                self.malicous_dns_df.append(temp_line[:2])
        self.malicous_dns_df=pd.DataFrame(self.malicous_dns_df,columns=['Family','Name'])
        self.normal_dns_df['is_dga']=0
        self.normal_dns_df['Family']='Normal'
        self.malicous_dns_df['is_dga']=1
        self.all_dns_df=pd.concat([self.normal_dns_df,self.malicous_dns_df],sort=False)
        self.all_dns_df['Subdomain'],self.all_dns_df['Domain'],self.all_dns_df['Suffix']=zip(*self.all_dns_df['Name'].map(self.extract_domain_info))
        self.dns_data_df = self.all_dns_df[['Domain','Suffix','Family','is_dga']]
        self.dns_data_df['Entropy']=self.dns_data_df['Domain'].map(self.entropy_calculator)
        self.dns_data_df['Lenth']=self.dns_data_df['Domain'].map(lambda x:len(x))
        self.dns_data_df['Vowel_ratio']=self.dns_data_df['Domain'].map(self.vowel_ratio)
        self.dns_data_df['Suffix_class']=self.dns_data_df['Suffix'].map(self.suffix_class)
        ngram_cv=feature_extraction.text.CountVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_df=1.0)
        document_matrix=ngram_cv.fit_transform(self.dns_data_df['Domain'])
        return document_matrix


    def build_model(self):
        pass
    
    

    def text(self):
        print(self.load_data())



if __name__ == "__main__":
    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows',None)

    dga=DGA_predictor()
    print(dga.text())