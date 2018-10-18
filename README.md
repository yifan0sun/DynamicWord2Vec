# DynamicWord2Vec
Paper title:
Dynamic Word Embeddings for Evolving Semantic Discovery. 

Paper links:
https://dl.acm.org/citation.cfm?id=3159703
https://arxiv.org/abs/1703.00607

Files:

/embeddings
 - embeddings in loadable MATLAB files. 0 corresponds to 1990, 1 to 1991, ..., 19 to 2009
 To save space, each year's embedding is saved separately. When used in visualization code, first merge to 1 embedding file.
 
/train_model
 - contains code used for training our embeddings
 - data file download: https://www.dropbox.com/s/tzkaoagzxuxtwqs/data.zip?dl=0
 
    /train_model/train_time_CD_smallnyt.py
     - main training script

    /train_model/util_timeCD.py
     - containing helper functions

/other_embeddings
 - contains code for training baseline embeddings
 - data file download: https://www.dropbox.com/s/tzkaoagzxuxtwqs/data.zip?dl=0
 
   /other_embeddings/staticw2v.py
    - static word2vec (Mikolov et al 2013)
    
   /other_embeddings/aw2v.py
    - aligned word2vec (Hamilton, Leskovec, Jufarsky 2016)
    
   /other_embeddings/tw2v.py
    - transformed word2vec (Kulkarni, Al-Rfou, Perozzi, Skiena 2015)
    
/visualization
 - scripts for visualizations in paper
 
   /visualization/norm_plots.py
    - changepoint detection figures
    
   /visualization/tsne_of_results.py
    - trajectory figures
    
/distorted_smallNYT
 - code for robust experiment
 - data file download: https://www.dropbox.com/s/6q5jhhmxdmc8n1e/data.zip?dl=0
 
/misc
 - contains general statistics and word hash file
