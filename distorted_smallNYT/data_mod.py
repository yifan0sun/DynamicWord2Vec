#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 20:23:29 2017

@author: raon
"""

'''
script to generate subsampled datasets etc
'''

import scipy.sparse as ss
import numpy as np
import pandas as pd
import numpy.random as nr


'''
/home/yaoz/DynamicWord2Vec/data/NYTimesV2/wordPairPMI_<0-27>.csv
'''


def yearmap(T):
    # list containing year numbers
    startyear = 1990
    endyear = startyear+len(T)
    return list(range(startyear,endyear))
    

def read_data(f,v):
    data = pd.read_csv(f)
    data = data.as_matrix()
    X = ss.coo_matrix((data[:,2],(data[:,0],data[:,1])),shape=(v,v))
    return X
    
def read_vocab(vfile):
    # list containing vocabulary
    vocab = []
    with open(vfile) as f:
        for line in f:
            vals = line.strip('\n').split(',')
            vocab.append(vals[1])
    return vocab    

def subsample(X,percentage):
    data = X.data
    rows = X.row
    cols = X.col
    N = len(cols)
    shap = X.shape
    shuf = list(nr.permutation(range(N)))
    cut  = int(np.floor(N*percentage))
    rows = rows[shuf[0:cut]]
    cols = cols[shuf[0:cut]]
    data = data[shuf[0:cut]]
    Y = ss.coo_matrix((data,(rows,cols)),shape=shap)
    return Y

def remove_word(X,wid):
    X = ss.csr_matrix(X)
    X[wid,:]=0
    X = ss.csc_matrix(X)
    X[:,wid]=0
    X = ss.coo_matrix(X)
    return X


def random_subsample_all(fhead,percentage):
    T = range(27)
    for t in T:
        f = fhead + str(t) + '.csv'
        X = read_data(f,v)
        Y = subsample(X,percentage)
        data = Y.data
        rows = Y.row
        cols = Y.col
        M = np.hstack((data,rows,cols))
        s = fhead + str(t) + '_S_' + str(percentage) + '.csv'
        np.savetxt(s,M,delimiter=',')
        print('{} of {}'.format(t,len(T)))
    return


def get_id(vocab,word):
    wid = vocab.index(word)
    return wid


    

if __name__=='__main__':
    
    fhead = '/Users/raon/Desktop/Projects_2016/GlobalEmbedding/Timeseries/data/wordPairPMI_'
#    fhead = '/home/yaoz/DynamicWord2Vec/data/NYTimesV2/wordPairPMI_'
    v = 20936
#    vocabfile = '/home/yaoz/DynamicWord2Vec/data/NYTimesV2/wordIDHash.csv'
    vocabfile = '/Users/raon/Desktop/Projects_2016/GlobalEmbedding/Timeseries/data/wordIDHash.csv'
    
    T = range(27)
    
    vocab =read_vocab(vocabfile)
    
    percentage = 0.2;
    random_subsample_all(fhead,percentage)
    print('subsampling {} done'.format(percentage))

    percentage = 0.8;
    random_subsample_all(fhead,percentage)
    print('subsampling {} done'.format(percentage))    
    
    
    word = 'apple'
    years= range(2010,2013)
    
    allyears = yearmap(T)
    wid = get_id(vocab,word)
    for year in years:
        t = T[allyears.index(year)]
        f = fhead + str(t) + '.csv'
        X = read_data(f,v)
        Y = remove_word(X,wid)
        s = fhead + str(t) + '_R_' + word + '.csv'
        np.savetxt(s,Y,delimiter=',')
        print('{} of {}'.format(year,len(years)))
        
    