# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 11:31:59 2017

@author: suny2

@descr: Creates AW2V embeddings loadable in matlab
"""

from scipy.spatial import procrustes
import scipy.io as sio
import numpy as np


embs = []
for k in xrange(180,200):
    embs.append(sio.loadmat('data/Uindep_%d.mat' % k)['U'])
    
for k in xrange(1,len(embs)):
    print k
    embs[k-1],embs[k],disp = procrustes(embs[k-1],embs[k])
    print disp
    
embs_dict = {}
for k in xrange(len(embs)):
    embs_dict['U_%d' % k] = embs[k]


sio.savemat('ngram_small/data/Aw2v.mat', embs_dict)




