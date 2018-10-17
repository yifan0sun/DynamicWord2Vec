# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:31:59 2017

@author: suny2
"""

from scipy.linalg import solve, lu, lu_solve
import scipy.io as sio
import numpy as np
from sklearn.neighbors import KDTree

years = range(180,200)
dim = 50
eff_dim_vec = [46,46,46,46,46,46,46,46,46,46,46,46,46,46,47,47,47,47,47,47] # computing eff_dim from get_eff_dim
def load_data():
    embs = []
    for k in years:
        embs.append(sio.loadmat('data/Uindep_%d.mat' % k)['U'])
    
    word2id = {}
    fid = open('data/wordlist.txt','r')
    k = 0
    for line in fid:
        word = line.strip('\n')
        word = line.strip('\r\n')
        word2id[word] = k
        k += 1
    fid.close()

    return embs, word2id
    
#checks to see effective dimension for each embedding, since some columns are all 0
def get_eff_dim(embs):
    for emb in embs:
        eff_dim = dim
        s = np.array(np.sum(np.multiply(emb,emb),axis=0)).ravel()
        while s[eff_dim-1] < .01: eff_dim -= 1
        print eff_dim
#Returns embedding at specified year, transformed to match the 10 closest words of the reference word in the reference year.
def get_emb( year, ref_word,ref_year):
    wid = word2id[ref_word]
    yid = years.index(year)
    ref_yid = years.index(ref_year)
    
    kdt = KDTree(embs[ref_yid], metric='euclidean')
    indices = kdt.query(embs[ref_yid][wid,:], k=10,return_distance = False)
    
    Uref = embs[ref_yid][indices,:]
    Utgt = embs[yid][indices,:]

    Uref, Utgt= np.matrix(Uref), np.matrix(Utgt)
    eff_dim = min(eff_dim_vec[yid], eff_dim_vec[ref_yid])
    
    Uref, Utgt = Uref[:,:eff_dim], Utgt[:,:eff_dim]
    W = np.zeros((dim,dim))
    W[:eff_dim,:eff_dim] = solve(np.dot(Utgt.T,Utgt), np.dot(Utgt.T,Uref))
    return np.dot(embs[ref_yid],W)


embs,word2id = load_data()

new_emb = get_emb(199, 'gay', 180)