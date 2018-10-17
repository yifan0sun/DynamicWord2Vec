#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:10:42 2016

@author: raon
"""

# main script for time CD 
# trainfile has lines of the form
# tok1,tok2,pmi

import numpy as np
import util_timeCD as util
import pickle as pickle
import scipy.io as sio
import copy

# PARAMETERS
''' T range for ngram is 180 -  199 
/home/yaoz/DynamicWord2Vec/data/googleNgram/pmis/pmi_<num>.txt

T range for nyt small is 0-27
/home/yaoz/DynamicWord2Vec/data/NYTimesV2/wordPairPMI_<num>.txt

'''
nw = 20936 # number of words in vocab (11068100/20936 for ngram/nyt)
T = range(1990,2017) # total number of time points (20/range(27) for ngram/nyt)
cuda = True

savehead = 'results/'
    
def print_params(r,lam,tau,gam,emph,ITERS):
    
    print('rank = {}'.format(r))
    print('frob  regularizer = {}'.format(lam))
    print('time  regularizer = {}'.format(tau))
    print('symmetry regularizer = {}'.format(gam))
    print('emphasize param   = {}'.format(emph))
    print('total iterations = {}'.format(ITERS))
    

def run(lam,gam,tau,ITERS,attack_years = [], savetail='', pmitail = ''):
    r   = 50  # rank
    b = nw # batch size
    emph = 1 # emphasize the nonzero
    savefile = savehead+'L'+str(lam)+'T'+str(tau)+'G'+str(gam)+'A'+str(emph)

    if len(savetail) > 0: savetail = '_' + savetail
    if len(pmitail) > 0: pmitail = '_' + pmitail
    savefile = savefile + savetail
    
    embfilename = 'data/emb_static%s' % savetail
    print savefile, embfilename
    try:
        e = sio.loadmat(embfilename)['emb']
    except(IOError):
        print 'file not available yet', embfilename
        return
        
    
    print('starting training with following parameters')
    print_params(r,lam,tau,gam,emph,ITERS)
    print('there are a total of {} words, and {} time points'.format(nw,T))
    
    print('X*X*X*X*X*X*X*X*X')
    print('initializing')
    
    Ulist = [copy.deepcopy(e) for x in T]
    Vlist = [copy.deepcopy(e) for x in T]
    del e
    
    
    print('getting batch indices')
    if b < nw:
        b_ind = util.getbatches(nw,b)
    else:
        b_ind = [range(nw)]
    
    import time
    start_time = time.time()
    # sequential updates
    for iteration in xrange(ITERS):  
        print_params(r,lam,tau,gam,emph,ITERS)
        try:
            Ulist = pickle.load(open( "%sngU_iter%d.p" % (savefile,iteration), "rb" ) )
            Vlist = pickle.load(open( "%sngV_iter%d.p" % (savefile, iteration), "rb" ) )
            print 'iteration %d loaded succesfully' % iteration
            continue
        except(IOError):
            pass
        # shuffle times
        if iteration == 0: times = T
        else: times = np.random.permutation(T)
        
        for t in xrange(len(times)):   # select a time
            print 'iteration %d, time %d' % (iteration, t)
            if T[t] in attack_years:
                f = 'data/wordPMI_%d%s.mat' % (T[t],pmitail)
            else:
                f = 'data/wordPMI_' + str(T[t]) + '.mat'
            
            pmi = sio.loadmat(f)['pmi']
            
            
            
            for j in xrange(len(b_ind)): # select a mini batch
                print '%d out of %d' % (j,len(b_ind))
                ind = b_ind[j]
                ## UPDATE V
                # get data
                pmi_seg = pmi[:,ind].todense()
                
                if t==0:
                    vp = np.zeros((len(ind),r))
                    up = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vp = Vlist[t-1][ind,:]
                    up = Ulist[t-1][ind,:]
                    iflag = False

                if t==len(T)-1:
                    vn = np.zeros((len(ind),r))
                    un = np.zeros((len(ind),r))
                    iflag = True
                else:
                    vn = Vlist[t+1][ind,:]
                    un = Ulist[t+1][ind,:]
                    iflag = False
                Vlist[t][ind,:] = util.update(Ulist[t],emph*pmi_seg,vp,vn,lam,tau,gam,ind,iflag)
                Ulist[t][ind,:] = util.update(Vlist[t],emph*pmi_seg,up,un,lam,tau,gam,ind,iflag)
            
      
                
            ####  INNER BATCH LOOP END
                
        # save
        print 'time elapsed = ', time.time()-start_time
       

        pickle.dump(Ulist, open( "%sngU_iter%d.p" % (savefile,iteration), "wb" ) , pickle.HIGHEST_PROTOCOL)
        pickle.dump(Vlist, open( "%sngV_iter%d.p" % (savefile, iteration), "wb" ) , pickle.HIGHEST_PROTOCOL)
if __name__=='__main__':
    attack_year_str = '1991-1994-1997-2000-2003-2006-2009-2012-2015'

    ITERS = 5 # total passes over the data
    lam = 10 #frob regularizer
    gam = 50 # forcing regularizer
    tau = 50  # smoothing regularizer  
    for rate in [0.1,0.01,0.001]:
        run(lam,gam,tau,ITERS,attack_years = [1991,1994,1997,2000,2003,2006,2009,2012,2015],savetail = 'rate%f_attackyear%s_all' %( rate,attack_year_str), pmitail = 'rate%f_all'%rate)
               
