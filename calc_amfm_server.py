#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
__author__ = 'luisdhe'
__version__ = "$Revision: 1.0.3 $"

# Common python modules
import os
import sys
import string
import json
import argparse
import cPickle as pickle
from lm import ArpaLM

try:
    import numpy as np
except:
    print "Error: Requires numpy from http://www.numpy.org/. Have you installed numpy?"
    sys.exit()

try:
    from sklearn.externals import joblib
except:
    print "Error: Requires sklearn from http://scikit-learn.org/. Have you installed scikit?"
    sys.exit()

try:
    from scipy.spatial.distance import cosine
except:
    print "Error: Requires scipy from http://scipy.org/. Have you installed scipy?"
    sys.exit()

# Socket module for handling the tcp connections
from socket import *
# Threads to handle multiple clients
from thread import *

# Defining server address and port
host = 'localhost'          # 'localhost' or '127.0.0.1' or '' are all same
MAX_CLIENTS = 5
DEFAULT_PORT = 52000

# Important directories for the system
root_dir = os.path.dirname(os.path.realpath(__file__))
models_dir = root_dir + '/models/'

# Global variables
CONF_VALUES = {
    'ASPEC_JP_EN':
        {
            'ROOT_DIR': 'ASPEC/en_jp',
            'jp': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train-1',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'en': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train-1',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'ASPEC_JP_ZH':
        {
            'ROOT_DIR': 'ASPEC/jp_zh',
            'jp': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 8000,  # Max size of the trained AM model
                'OPT_AM_SIZE': 4000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'zh': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 8000,  # Max size of the trained AM model
                'OPT_AM_SIZE': 4000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },

    'JPO_JP_ZH':
        {
            'ROOT_DIR': 'JPO_PATENT/jp_zh',
            'jp': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'zh': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'JPO_JP_EN':
        {
            'ROOT_DIR': 'JPO_PATENT/jp_en',
            'jp': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'en': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'JPO_JP_KO':
        {
            'ROOT_DIR': 'JPO_PATENT/jp_ko',
            'jp': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'ko': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'BPPT_IN_EN':
        {
            'ROOT_DIR': 'BPPT/in_en',
            'in': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'en': {
                'NUM_TRAIN_SENT': 10000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'HINDEN_HI_EN':
        {
            'ROOT_DIR': 'HINDEN/hi_en',
            'hi': {
                'NUM_TRAIN_SENT': 7000,  # Number of sentences used during training
                'FULL_AM_SIZE': 1500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 1500,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'IITB.hi-en',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'en': {
                'NUM_TRAIN_SENT': 7000,  # Number of sentences used during training
                'FULL_AM_SIZE': 1500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 1500,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'IITB.hi-en',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'WAT2016_JP_HI':
        {
            'ROOT_DIR': 'WAT2016/jp_hi',
            'hi': {
                'NUM_TRAIN_SENT': 20000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'jp': {
                'NUM_TRAIN_SENT': 20000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'JIJI_EN_JP':
        {
            'ROOT_DIR': 'JIJI/en_jp',
            'en': {
                'NUM_TRAIN_SENT': 20000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 500,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'jp': {
                'NUM_TRAIN_SENT': 20000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 1000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'RECIPES_EN_JP_TITLES':
        {
            'ROOT_DIR': 'RECIPES/en_jp',
            'en': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 100,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_titles',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'jp': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 100,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_titles',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'RECIPES_EN_JP_STEPS':
        {
            'ROOT_DIR': 'RECIPES/en_jp',
            'en': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 2000,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_steps',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'jp': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 1200,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_steps',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'RECIPES_EN_JP_INGREDIENTS':
        {
            'ROOT_DIR': 'RECIPES/en_jp',
            'en': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 80,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_ingredients',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'jp': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 500,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_ingredients',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'RECIPES_EN_JP_ALL':
        {
            'ROOT_DIR': 'RECIPES/en_jp',
            'en': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 550,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_all',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'jp': {
                'NUM_TRAIN_SENT': 14000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 750,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train_all',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
    'MY_EN_ALL':
        {
            'ROOT_DIR': 'WAT18MY-EN/my_en',
            'en': {
                'NUM_TRAIN_SENT': 13000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 550,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train.alt',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
            'my': {
                'NUM_TRAIN_SENT': 13000,  # Number of sentences used during training
                'FULL_AM_SIZE': 2500,  # Max size of the trained AM model
                'OPT_AM_SIZE': 750,  # Optimal value for the trained AM model
                'NGRAM_ORDER': 3,  # Order the FM score calculation
                'NUM_FOLD': 0,  # Fold used to train the models
                'PREFIX_AM_FM': 'train.alt',  # Prefix for the AM-FM models
                'ALPHA': 0.5  # Interpolation value for AM-FM
            },
        },
}

sc = set(['-', "'", '%'])
to_remove = ''.join([c for c in string.punctuation if c not in sc])
table = dict((ord(char), u'') for char in to_remove)


sc = set([',', '!', '?', '.'])
to_separate = ''.join([c for c in string.punctuation if c not in sc])
table_separate = dict((ord(char), u' ' + char) for char in to_separate)


# Implementation of the vector space model
class VSM:
    def __init__(self, model_file, size_am):
        self.am = None
        self.vectorizer = None
        self.load(model_file)
        self.am_components = self.am[:, 0:size_am]
        self.cache_refvectors = dict()

    # Function to get the distance between a set of reference and test sentences
    def search(self, ref_sentence, test_sentence):
        """ search for documents that match based on a list of terms """
        reference_vector = self.vectorizer.transform([ref_sentence])
        target_vector = self.vectorizer.transform([test_sentence])

        if ref_sentence not in self.cache_refvectors:
            ref = reference_vector.dot(self.am_components)
            self.cache_refvectors[ref_sentence] = ref
        else:
            ref = self.cache_refvectors[ref_sentence]

        if test_sentence not in self.cache_refvectors:
            tgt = target_vector.dot(self.am_components)
            self.cache_refvectors[test_sentence] = tgt
        else:
            tgt = self.cache_refvectors[test_sentence]

        return max(0.0, 1.0 - cosine(ref, tgt))  # Avoid sending negative distances

    # Load models
    def load(self, name_model):
        print('Loading AM model')
        self.am = joblib.load(name_model + '.h5')
        file_h = open(name_model + '.dic', "rb")
        self.vectorizer = pickle.load(file_h)
        file_h.close()


class calcScoresAMFM:
    def __init__(self, dataset, lang='en', am=True, fm=True):
        # Load configuration variables for language
        self.DATASET_DIR = CONF_VALUES[dataset]['ROOT_DIR']
        self.FULL_AM_SIZE = CONF_VALUES[dataset][lang]['FULL_AM_SIZE']
        self.OPT_AM_SIZE = CONF_VALUES[dataset][lang]['OPT_AM_SIZE']
        self.NUM_TRAINING_SIZE = CONF_VALUES[dataset][lang]['NUM_TRAIN_SENT']
        self.PREFIX_AM_FM = CONF_VALUES[dataset][lang]['PREFIX_AM_FM']
        self.NGRAM_ORDER = CONF_VALUES[dataset][lang]['NGRAM_ORDER']
        self.NUM_FOLD = CONF_VALUES[dataset][lang]['NUM_FOLD']
        self.alpha = CONF_VALUES[dataset][lang]['ALPHA']
        self.lang = lang
        self.am = am
        self.fm = fm
        self.cache_lm = dict()  # Store previously calculated n-gram values for speed

        if self.am is True:
            # Check that the AM models exist
            am_full_matrix = models_dir + '/' + self.DATASET_DIR + '/' + self.PREFIX_AM_FM + '.' + lang + '.' \
                             + str(self.NUM_TRAINING_SIZE) + \
                             '.' + str(self.FULL_AM_SIZE) + '.' + str(self.NUM_FOLD)
            if not os.path.isfile(am_full_matrix + '.h5') or not os.path.isfile(am_full_matrix + '.dic'):
                print ('******* ERROR: files: ' + am_full_matrix + '.h5 or ' + am_full_matrix + '.dic does not exists.')
                exit()
            elif os.path.getsize(am_full_matrix + '.h5') == 0 or os.path.getsize(am_full_matrix + '.dic') == 0:
                print ('******* ERROR: Check if files: ' + am_full_matrix + '.h5 or ' + am_full_matrix +
                       '.dic are not empty.')
                exit()

        print('Starting loading models for language %s ...' % (lang))
        if self.fm is True:
            # Check that the LM model exists
            lm_model = models_dir + '/' + self.DATASET_DIR + '/' + self.PREFIX_AM_FM + '.' + lang + '.' + str(self.NGRAM_ORDER) + '.lm'
            if not os.path.exists(lm_model):
                print("******* ERROR: LM file " + lm_model + ' does not exists.')
                exit()
            elif os.path.getsize(lm_model) == 0:
                print("******* ERROR: LM file " + lm_model + ' is empty.')
                exit()
            print('Loading FM model...')
            self.lm = ArpaLM(lm_model)

        if self.am is True:
            # Load the models
            self.vs = VSM(am_full_matrix, self.OPT_AM_SIZE)

        print('Finished loading models for language %s ...' % (lang))

    # Perform basic pre-processing applied during training
    def doProcessFromStrings(self, ref, pred):
        ref = self.preProcess(ref, self.lang)
        pred = self.preProcess(pred, self.lang)
        return ref, pred

    # Pre-Processing for each sentence. In the case of languages different to English we perform tokenization
    # per character
    def preProcess(self, s, lang):
        if len(s) == 0:  # To avoid empty lines
            return '_EMPTY_'

        # Remove some punctuation
        s = s.translate(table)
        if lang == 'my':  # we need to remove separation for | and -
            for k, v in table_separate.iteritems():
                if v == u' |' or v == u' -':
                    table_separate[k] = v.strip()
        s = s.translate(table_separate)

        # Tokenization by characters for most of Asian languages except for English, Indian, Korean and Myanmar
        if lang != 'en' and lang != 'in' and lang != 'ko' and lang != 'hi' and lang != 'my':
            tokens = [' '.join([c for c in list(word.strip())]) for word in s.split()]
        else:
            tokens = s.split()
            if lang == 'my':  # We need to replace the | for -
                tokens = [x.replace('|', '-') for x in tokens]

        s = ' '.join(tokens).lower()
        return s

    # Function to calculate the FM metric using language models
    def calculateFMMetric(self, ref, tst):
        if self.lang != 'en'and self.lang != 'in' and self.lang != 'ko' and self.lang != 'hi' and self.lang != 'my':
            ref = ' '.join(list(ref.strip()))
            tst = ' '.join(list(tst.strip()))

        if self.lang == 'my':  # We need to replace the | for -
            ref = ref.replace('|', '-')
            tst = tst.replace('|', '-')

        sent = '<s> ' + ref.strip() + ' </s>'
        aWords = sent.split()
        num_words_ref = len(aWords) - 2
        prob_ref = 0.0
        # Calculates the log-prob for the different n-grams
        for i in range(1, len(aWords)):
            # prob_ref += self.lm.score(tuple(aWords[max(0, i - self.NGRAM_ORDER + 1):i + 1]))
            words = aWords[max(0, i - self.NGRAM_ORDER + 1):i + 1]
            ngram = ' '.join(words)
            # Try to speed calculation by using cache values
            try:
                prob_ref += self.cache_lm[ngram]
            except:
                val = self.lm.score(tuple(words))
                self.cache_lm[ngram] = val
                prob_ref += val

        sent = '<s> ' + tst.strip() + ' </s>'
        aWords = sent.split()
        num_words_tst = len(aWords) - 2
        prob_tst = 0.0
        # Calculates the log-prob for the different n-grams
        for i in range(1, len(aWords)):
            # prob_tst += self.lm.score(tuple(aWords[max(0, i-self.NGRAM_ORDER+1):i+1]))
            words = aWords[max(0, i - self.NGRAM_ORDER + 1):i + 1]
            ngram = ' '.join(words)
            # Try to speed calculation by using cache values
            try:
                prob_tst += self.cache_lm[ngram]
            except:
                val = self.lm.score(tuple(words))
                self.cache_lm[ngram] = val
                prob_tst += val


        # Calculate the scaled probability
        prob_ref = np.exp(prob_ref / num_words_ref)
        prob_tst = np.exp(prob_tst / num_words_tst)
        return 1.0 - ((max(prob_tst, prob_ref) - min(prob_tst, prob_ref))/max(prob_tst, prob_ref))


    # Functionality to calculate the AM score using monolingual SVM
    def calculateAMMetric(self, ref, pred):
        return self.vs.search(ref, pred)


def processSubmission(target, submission, cs, fm, am):
        (target, submission) = cs.doProcessFromStrings(ref=target, pred=submission)

        res_fm = -1.0
        if fm is True:
            res_fm = cs.calculateFMMetric(target, submission)

        res_am = -1.0
        if am is True:
            res_am = min(1.0, cs.calculateAMMetric(target, submission))

        res_am_fm = -1.0
        if am is True and fm is True:
            res_am_fm = cs.alpha * res_am + (1.0 - cs.alpha) * res_fm

        return (res_am_fm, res_am, res_fm, cs.alpha)


def clientthread(conn, cs):
    while True:
        # Receiving from client
        data = conn.recv(16384)
        info_message = json.loads(data)
        if info_message['data'] == 'finish':
            break

        ref = info_message['ref']
        out = info_message['out']
        am = bool(info_message['am'])
        fm = bool(info_message['fm'])
        lang = info_message['lang']
        if cs.lang != lang:
            msg = 'Sorry, but you are trying to evaluate a submission for language: %s, but the server is waiting for ' \
                  'language: %s. The client must be disconnected' % (lang, cs.lang)
            print (msg)
            out_message = dict()
            out_message['data'] = 'finish'
            out_message['err_msg'] = msg
            conn.send(json.dumps(out_message))
            break
        res_am_fm, res_am, res_fm, alpha = processSubmission(ref, out, cs, am, fm)

        out_message = dict()
        out_message['am_fm'] = res_am_fm
        out_message['am'] = res_am
        out_message['fm'] = res_fm
        out_message['alpha'] = alpha
        conn.send(json.dumps(out_message))
    conn.close()
    print ('Connection closed with client')


def main():

    valid_datasets = []  # [ASPEC_JP_ZH, ASPEC_JP_EN, JPO_JP_ZH, JPO_JP_EN, JPO_JP_KO, BPPT_IN_EN, HINDEN_IN_EN]
    for k in CONF_VALUES:
        valid_datasets.append(k)

    parser = argparse.ArgumentParser()
    parser.add_argument("-port", "--port", default=DEFAULT_PORT, help="Port used to listen to the clients "
                                                                      "[default: 52000]")
    parser.add_argument("dataset", help="Dataset prefix [" + ', '.join(valid_datasets) + ']')
    parser.add_argument("lang", help="Target language [en|jp]")

    args = parser.parse_args()

    # Check integrity of given dataset and language
    if args.dataset not in valid_datasets:
        print ('ERROR: the specified dataset (%s) is not valid. Only these are accepted: %s' %(args.dataset,
                                                                                               ', '.join(valid_datasets)
                                                                                               ))
        exit()

    if args.lang not in CONF_VALUES[args.dataset]:
        print ('ERROR: the specified language (%s) is not defined for the given dataset (%s), valid values are: %s' %
               (args.lang, args.dataset, ', '.join([l for l in CONF_VALUES[args.dataset] if len(l) == 2])))
        exit()

    # Creating socket object
    sock = socket()
    # Binding socket to a address. bind() takes tuple of host and port.
    sock.bind((host, int(args.port)))
    # Listening at the address
    sock.listen(MAX_CLIENTS)  # 5 denotes the number of clients can queue
    print ('Listening clients on %s:%s' %(host, args.port))
    # Load the models for AM and FM
    cs = calcScoresAMFM(dataset=args.dataset, lang=args.lang, am=True, fm=True)
    print ('Waiting for connections...')

    while True:
        # Accepting incoming connections
        conn, addr = sock.accept()
        print ('Connection accepted with client')
        # Creating new thread. Calling clientthread function for this function and passing conn as argument.
        start_new_thread(clientthread, (conn, cs))  # start new thread takes 1st argument as a function name to be run,
        # second is the tuple of arguments to the function.

    conn.close()
    sock.close()


if __name__ == '__main__':
    main()