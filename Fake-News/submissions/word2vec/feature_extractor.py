# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import logging
logger = logging.getLogger(__name__)

import numpy as np
import string
import unicodedata

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler

from gensim.models import Word2Vec

import pandas as pd
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

import os
import sys
import tarfile
from gensim.models.keyedvectors import KeyedVectors
import collections as coll


from itertools import izip,tee 
from nltk.corpus import stopwords,wordnet
from nltk.tag import pos_tag
from gensim.models.keyedvectors import KeyedVectors

##################### Download WORD2VEC google



def download_file(path,url):
    ''' 
    downlad file from the url path to the specified path
    '''
    import urllib
    testfile = urllib.URLopener()
    testfile.retrieve(url,path)

def unzip(path):
    '''
    unzips a file and returns the filenames of the unzipped items
    '''
    print path
    tar=tarfile.open(path)
    file_names = tar.getnames()
    for file_name in file_names:
        tar.extract(file_name,f_name)
    tar.close()
    return file_names
    
def file_exists(path):
    '''
    checks if a file exists
    '''
    if (os.path.isfile(path)):
        return True
    else:
        return False


# https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

var_exists = 'WORD2VEC_GLOBAL' in locals() or 'WORD2VEC_GLOBAL' in globals()

if(var_exists):
    print("Already W2C loaded")
else:

    print("!!!!!!!!!!!! Loading W2C into RAM !!!!!!!!!!!")
    new_path = os.path.join(
        os.getcwd(),
        'data' # attention not this on the server !!!
        )
    NAME = 'GoogleNews-vectors-negative300.bin'
    NAME_zip = 'GoogleNews-vectors-negative300.bin'


    data = os.path.join(
    new_path,
    NAME
    )
    data_zip = os.path.join(
    new_path,
    NAME_zip
    )

    if(file_exists(data) == False):
    # do nothing
        url = ' http://s3-us-west-2.amazonaws.com/awse***/GoogleNews-vectors-negative300.bin'  # Private URL (limited bandwidth $$), to re-use my code, please use :https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz 
        download_file(data_zip,url)
    # Unzip
        
        #unzip(data_zip)
        WORD2VEC_GLOBAL =  KeyedVectors.load_word2vec_format(data, binary=True)
    else:
    # download file
        print("File exists")
        data = os.path.join(
        new_path,
        NAME
        )

        WORD2VEC_GLOBAL =  KeyedVectors.load_word2vec_format(data, binary=True)

#####################


def pairwise_tokenize(sentence,w2v,remove_stopwords=True):
    
    ignore_words = stopwords.words('english')

    #Remove non-alphanumeric
    pattern = re.compile('[\W_]+') 
    sentence = pattern.sub(" ",sentence)  
    sentence = sentence.strip()
    words = sentence.split(" ")

    compound_word_idx = []
    a_idx = 0
    for a,b in pairwise(words):
        combined = a +"_" + b
        try:
            w2v[combined]
            compound_word_idx.append(a_idx) #append the index of the 1st compound word
            a_idx += 1
        except KeyError:
            a_idx += 1

    for idx in compound_word_idx:
        words[idx] = words[idx] + "_" + words[idx + 1] #compound_word_idx stores index of 1st word, so combine with the next word

    #This cannot be combined into another loop to maintain where indices point
    for idx in reversed(compound_word_idx):
        words.pop(idx+1)

    if remove_stopwords == True:
        filtered = []
        for word in words:
            word = word.decode("utf-8")
            if word not in ignore_words:
                filtered.append(word)

        words = filtered

    return words



def sentence_to_mat(sentence,w2vmodel):
    temp = []
    words = pairwise_tokenize(sentence,w2vmodel,remove_stopwords=True)
    for j in words:
        try:
            temp.append(w2vmodel[j].reshape((300,1)))
        except KeyError:
            continue
    temp = np.concatenate(temp, axis=1)
    return temp
    return words



def directory_exists(path):
    '''
    checks if a directory exists
    '''
    if (os.path.isdir(path)):
        return False
    else:
        return True


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)



def prep_tensor_data(headlines, w2v):


    X = []
    #y = []
    #headlines.reset_index(drop=True)

    for i in range(headlines.shape[0]):
        #print(i)
        #print("!!!!!!!!!!!")
        #print(headlines.shape[0])
        #print(i)

        head = sentence_to_mat(headlines.iloc[i], w2v)  # .iloc (if not changed in cleaning string)
        head_avg = np.mean(head, axis=1)

        #bod = sentence_to_mat(bodies[i],w2v)
        #bod_avg = np.mean(bod, axis=1)

        U_head, S_head, V_head = np.linalg.svd(head)
        mat_1 = U_head[:,:0]
        #U_bod, S_bod, V_bod = np.linalg.svd(bod)
        #mat_2 = U_bod[:,:7]

        tmp = np.concatenate((head_avg.reshape((head_avg.shape[0],1)), mat_1), axis=1)
        #tmp = mat_1
        X.append(tmp)
        #y.append(Y[i])
        
    return np.dstack(X).T #, np.hstack(y)


import unicodedata


def clean_str(sentence, stem=True):
    english_stopwords = set(
        [stopword for stopword in stopwords.words('english')])
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    if stem:
        stemmer = SnowballStemmer('english')
        return list((filter(lambda x: x.lower() not in english_stopwords and
                            x.lower() not in punctuation,
                            [stemmer.stem(t.lower())
                             for t in word_tokenize(sentence)
                             if t.isalpha()])))

    return list((filter(lambda x: x.lower() not in english_stopwords and
                        x.lower() not in punctuation,
                        [t.lower() for t in word_tokenize(sentence)
                         if t.isalpha()])))


def strip_accents_unicode(s):
    try:
        s = unicode(s, 'utf-8')
    except NameError:  # unicode is a default on python 3
        pass
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)




class FeatureExtractor():

    def __init__(self):

        self.size = 300
        self.name = 'GoogleNews-vectors-negative300.bin'
        self.url = "Test" # Add download from a link !!!
        self.w2v = WORD2VEC_GLOBAL;
        self.text = None 
        self._matrix = None
        self._feat = None


    def fit(self, X_df, y):

        My_chain = (X_df.statement) + X_df.source.astype(str) + X_df.state.astype(str) + X_df.subjects.astype(str)
        self._feat = My_chain #np.array([' '.join(clean_str(strip_accents_unicode(dd))) for dd in My_chain]) # My_chain 
        
        return self._feat
        

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self.transform(X_df) 


    def transform(self, X_df):
        My_chain = (X_df.statement) + X_df.source.astype(str) + X_df.state.astype(str) + X_df.subjects.astype(str)
        self._matrix = prep_tensor_data(My_chain, self.w2v)
        self._matrix = self._matrix.reshape([self._matrix.shape[0],self._matrix.shape[2]])
        return self._matrix
