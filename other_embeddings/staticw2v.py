# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 13:17:12 2017

@author: suny2
"""
import gensim
import scipy.io as sio
import numpy as np

pmi = sio.loadmat('data/pmi_all.mat')['pmi_all'].todense()

model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
