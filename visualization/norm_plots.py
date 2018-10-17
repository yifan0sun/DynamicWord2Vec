# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 10:56:59 2017

@author: suny2
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:46:52 2017

@author: suny2
"""
import scipy.io as sio
import numpy as np
import json
from pprint import pprint

#%%

wordlist = []
fid = open('data/wordlist.txt','r')
for line in fid:
    wordlist.append(line.strip())
fid.close()
nw = len(wordlist)
    
word2Id = {}
for k in xrange(len(wordlist)):
    word2Id[wordlist[k]] = k
times = range(180,200) # total number of time points (20/range(27) for ngram/nyt)
emb_all = sio.loadmat('results/emb_frobreg10_diffreg50_symmreg10_iter10.mat')

#%%
words = ['thou','chaise','darwin','telephone']
allnorms = []
for w in words:
    norms = []
    for year in times:
        emb = emb_all['U_%d' % times.index(year)][word2Id[w],:]
        norms.append(np.linalg.norm(emb))
    
    norms = np.array(norms)
    norms = norms / sum(norms)
    allnorms.append(norms)
#%%
import matplotlib.pyplot as plt
import pickle
#Z = sio.loadmat('tsne_output/%s_tsne.mat'%word)['emb']
#list_of_words = pickle.load(open('tsne_output/%s_tsne_wordlist.pkl'%word,'rb'))
years = [t*10 for t in times]
markers = ['+','o','x','*']
plt.clf()
for k in xrange(len(allnorms)):
    norms = allnorms[k]
    plt.plot(years,norms,marker=markers[k],markersize=7)
plt.legend(words)
plt.xlabel('year')
plt.ylabel('word norm')
plt.show()

#sio.savemat('tsne_output/%s_tsne.mat'%word,{'emb':Z})
#pickle.dump(list_of_words,open('tsne_output/%s_tsne_wordlist.pkl'%word,'wb'))