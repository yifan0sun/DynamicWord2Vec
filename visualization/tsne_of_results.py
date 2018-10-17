# -*- coding: utf-8 -*-
"""
Created on Mon Jan 09 20:46:52 2017

@author: suny2
"""
import scipy.io as sio
import numpy as np
import json
from pprint import pprint
from scipy.spatial.distance import pdist
#%%


word = 'communist'


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

nn = 50
nc = 5
emb = emb_all['U_%d' % times.index(199)]
             
X = []
list_of_words = []
isword = []
v = emb[word2Id[word],:]
for year in times:


    emb = emb_all['U_%d' % times.index(year)]
    embnrm = np.reshape(np.sqrt(np.sum(emb**2,1)),(emb.shape[0],1))
    emb_normalized = np.divide(emb, np.tile(embnrm, (1,emb.shape[1])))           
    print emb_normalized.shape
    v = emb_normalized[word2Id[word],:]




    d =np.dot(emb_normalized,v)
    
    idx = np.argsort(d)[::-1]
    newwords = [(wordlist[k], year) for k in list(idx[:nn])]
    print newwords
    list_of_words.extend(newwords)
    for k in xrange(nn):
        isword.append(k==0)
    X.append(emb[idx[:nn],:])
    #print year, [wordlist[i] for i in idx[:nn]]
    
X = np.vstack(X)

print X.shape

#%%

from sklearn.manifold import TSNE
model = TSNE(n_components=2, metric = 'euclidean')
Z = model.fit_transform(X)


#%%



allwords = ['art','damn','gay','hell','maid','muslim']

import matplotlib.pyplot as plt
import pickle

plt.clf()
traj = []
for k in xrange(len(list_of_words)):
    
    if isword[k] :
        marker = 'ro'
        traj.append(Z[k,:])
    else: marker = 'b.'
    
    
    plt.plot(Z[k,0], Z[k,1],marker)
    plt.text(Z[k,0], Z[k,1],list_of_words[k])

traj = np.vstack(traj)
plt.plot(traj[:,0],traj[:,1])
plt.show()

sio.savemat('tsne_output/%s_tsne.mat'%word,{'emb':Z})
pickle.dump({'words':list_of_words,'isword':isword},open('tsne_output/%s_tsne_wordlist.pkl'%word,'wb'))

#%%
allwords = ['art','damn','gay','hell','maid','muslim']

import matplotlib.pyplot as plt
import pickle
Z = sio.loadmat('tsne_output/%s_tsne.mat'%word)['emb']
data = pickle.load(open('tsne_output/%s_tsne_wordlist.pkl'%word,'rb'))
list_of_words, isword = data['words'],data['isword']
plt.clf()
traj = []


Zp = Z*1.
Zp[:,0] = Zp[:,0]*2.
all_dist = np.zeros((Z.shape[0],Z.shape[0]))
for k in xrange(Z.shape[0]):
    all_dist[:,k] =np.sum( (Zp - np.tile(Zp[k,:],(Z.shape[0],1)))**2.,axis=1)

dist_to_centerpoints = all_dist[:,isword]
dist_to_centerpoints = np.min(dist_to_centerpoints,axis=1)

dist_to_other = all_dist + np.eye(Z.shape[0])*1000.
idx_dist_to_other = np.argsort(dist_to_other,axis=1)
dist_to_other = np.sort(dist_to_other,axis=1)

plt.clf()
for k in xrange(len(list_of_words)-1,-1,-1):
    
    if isword[k] :
        #if list_of_words[k][1] % 3 != 0 and list_of_words[k][1] < 199 : continue
        marker = 'bo'
        traj.append(Z[k,:])
        plt.plot(Z[k,0], Z[k,1],marker)
    else: 
        if dist_to_centerpoints[k] > 200: continue
        skip =False
        for i in xrange(Z.shape[0]):
            if dist_to_other[k,i] < 150 and idx_dist_to_other[k,i] > k: 
                skip = True
                break
            if dist_to_other[k,i] >= 150: break
        
        if skip: continue
        if Z[k,0] > 8: continue
        plt.plot(Z[k,0], Z[k,1])
    
    
    plt.text(Z[k,0]-2, Z[k,1]+np.random.randn()*2,' %s-%d' % (list_of_words[k][0], list_of_words[k][1]*10))

plt.axis('off')
traj = np.vstack(traj)
plt.plot(traj[:,0],traj[:,1])
plt.show()
