#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'vanmaren'
__version__ = "$Revision: 0.0.0"

# Common python modules
import os, sys, string
import argparse
import configargparse
import yaml
import pickle
from sklearn.externals import joblib
from scipy.spatial.distance import cosine
from lm import ArpaLM
import numpy as np
import preProcessModuleForTest as pp
import joblib as jlib


def processSubmissionNew(target, submission, cs, fm, am, cfg):
    """Initial method that will start the process of calculating the AMFM scores
        Args:
            target: refers to the path for the file that you will use as your reference for AMFM
            submission: refers to the path of the file that we use as submission ( this is output from and SMT e.g: GoogleTraductor)
            cs: refers to the class of calcScoresAMFM
            fm: boolean that refers whether we want to use FM metric True or not False
            am: boolean that refers whether we want to use AM metric True or not False
            cfg:refers to the configuration file that we use to configure the AMFM parameters (this depends mostly on the models that we are using for test
    """
    results = []
    alpha = cs.alpha

    if (cfg['test']['preProcess'] == True):
        (target, submission) = cs.doProcessFromStrings(ref=target, pred=submission)

    with open(target, 'r') as r1, open(submission, 'r') as r2:
        for line_target, line_submission in zip(r1, r2):
            res_fm = -1.0
            if fm is True:
                res_fm = cs.calculateFMMetric(line_target, line_submission)
                res_am = -1.0
            if am is True:
                res_am = min(1.0, cs.calculateAMMetric(line_target, line_submission))
                res_am_fm = -1.0
            if am is True and fm is True:
                res_am_fm = cs.alpha * res_am + (1.0 - cs.alpha) * res_fm
                res = (res_am_fm, res_am, res_fm, cs.alpha)
                results.append((res_fm, res_am, res_am_fm))

    print('N_SENT,\tFM,\tAM,\tAM_FM')
    for num, line in enumerate(results):
        print('%i,\t%.5f,\t%.5f,\t%.5f  ' % (num + 1, line[0], line[1], line[2]))
        # (num+1, results(num+1), results(numa+2), results(num+3))

    # Calculate FM global score
    if fm is True:
        print('GLOBAL AVERAGE FM: %.5f' % np.average([r[0] for r in results]))

    # Calculate AM global score
    if am is True:
        print('GLOBAL AVERAGE AM: %.5f' % np.average([r[1] for r in results]))

    # Calculate Interpolated AM_FM score
    if am is True and fm is True:
        print('GLOBAL AVERAGE AM_FM (%.2f): %.5f' % (alpha, np.average([r[2] for r in results])))

    print('********* END PROCESSING SUBMISSION ************\n')


class calcScoresAMFM:
    """Main classs that is used for the process of calculationg the scores """

    def __init__(self, cfg, lang='en', am=True, fm=True):
        # Load configuration variables for language
        self.cfg = cfg
        self.DATASET_DIR = cfg['test']['DATASET_DIR']
        self.FULL_AM_SIZE = cfg['test']['FULL_AM_SIZE']
        self.OPT_AM_SIZE = cfg['test']['OPT_AM_SIZE']
        self.NUM_TRAINING_SIZE = cfg['test']['NUM_TRAINING_SIZE']
        self.PREFIX_AM_FM = cfg['test']['PREFIX_AM_FM']
        self.NGRAM_ORDER = cfg['test']['NGRAM_ORDER']
        self.NUM_FOLD = cfg['test']['NUM_FOLD']
        self.alpha = cfg['test']['alpha']
        self.lang = lang
        self.am = am
        self.fm = fm
        self.cache_lm = dict()  # Store previously calculated n-gram values for speed
        self.models_dir = cfg['test']['ModelsDir']
        self.TestDir= cfg['test']['TestDir']

        self.sc = set(['-', "'", "%"])
        self.to_remove = ''.join([c for c in string.punctuation if c not in self.sc])
        self.to_remove = self.to_remove + "“" + "”"  # We had some extra characters to remove from texts
        self.table = dict((ord(char), u' ') for char in
                          self.to_remove)  # We replace the characters in the list(strg) with a blanc space = u' ' ( unicode blanc space)

        self.sc = set([',', '!', '?', '.'])
        self.to_separate = ''.join([c for c in string.punctuation if c not in self.sc])
        self.table_separate = dict((ord(char), u' ' + char + u' ') for char in
                                   self.to_separate)  # We separate the characters in the list(strg) with a blanc space = u' ' ( unicode blanc space)


        if self.am is True:
            # Check that the AM models exist
            am_full_matrix = self.models_dir + '/' + self.DATASET_DIR + '/' + self.PREFIX_AM_FM + '.' + lang + '.' \
                             + str(self.NUM_TRAINING_SIZE) + \
                             '.' + str(self.FULL_AM_SIZE) + '.' + str(self.NUM_FOLD)
            if not os.path.isfile(am_full_matrix + '.h5') or not os.path.isfile(am_full_matrix + '.dic'):
                print('******* ERROR: files: ' + am_full_matrix + '.h5 or ' + am_full_matrix + '.dic does not exists.')
                exit()
            elif os.path.getsize(am_full_matrix + '.h5') == 0 or os.path.getsize(am_full_matrix + '.dic') == 0:
                print('******* ERROR: Check if files: ' + am_full_matrix + '.h5 or ' + am_full_matrix +
                      '.dic are not empty.')
                exit()

        print('Starting loading models for language %s ...' % (lang))
        if self.fm is True:
            # Check that the LM model exists
            lm_model = self.models_dir + '/' + self.DATASET_DIR + '/' + self.PREFIX_AM_FM + '.' + lang + '.' + str(
                self.NGRAM_ORDER) + '.lm'
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

    def doProcessFromStrings(self, ref, pred):
        """ Perform basic pre-processing applied during training """
        ref = self.preProcess(ref, self.lang)
        pred = self.preProcess(pred, self.lang)
        return ref, pred

    def preProcess(self, s, lang):
        """ Pre-Processing for each sentence. In the case of languages different to English we perform tokenization per character
            parameters:
            s: referes to the file that we want to preprocess
            lang: refers to the language that we will use for AMFM scoring
        """

        # Creates the specfic File in "Files_input" for storing all the output files related to each Language
        if not os.path.exists('preProcessed'):
            print("...creating " + 'preProcessed')
            os.makedirs('preProcessed')

        decompositionOfThePath = s.split('/')
        filename = decompositionOfThePath[len(decompositionOfThePath)-1] #we extract the filename

        with open(s, 'r') as f_in, open(self.TestDir+'/preProcessed/'+filename, 'w+',encoding='utf-8') as f_out:
            for line in f_in:  # f_in.readlines():
                cP = pp.PreProcessingClass()
                s = cP.preProcessWork(line,lang)
                s = s.lower()

                f_out.write(s + '\n')
        return f_out.name

    def calculateFMMetric(self, ref, tst):
        """Function to calculate the FM metric using language models"""
        if self.lang != 'en' and self.lang != 'in' and self.lang != 'ko' and self.lang != 'hi' and self.lang != 'my':
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
        print(prob_ref)
        print(prob_tst)
        # return 1.0 - ((max(prob_tst, prob_ref) - min(prob_tst, prob_ref))/max(prob_tst, prob_ref))
        return min(prob_tst, prob_ref) / max(prob_tst, prob_ref)

    # Functionality to calculate the AM score using monolingual SVM
    def calculateAMMetric(self, ref, pred):
        return self.vs.search(ref, pred)

class VSM:
    """ class that holds the Implementation of the vector space model """
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

        #return max(0.0, 1.0 - cosine(ref, tgt))  # Avoid sending negative distances
        return 1.0 - cosine(ref, tgt)
    # Load models
    def load(self, name_model):
        # WAT2019: added because incompatibilities when reading old files created using python2
        try:
            self.am = jlib.load(name_model + '.h5')
        except:
            try:
                self.am = joblib.load(name_model + '.h5')
            except:
                file_h = open(name_model + '.h5', "rb")
                self.am = pickle.load(name_model + '.h5')
                file_h.close()

        # WAT2019: added because incompatibilities when reading old files created using python2
        try:
            self.vectorizer = jlib.load(name_model + '.dic')
        except:
            try:
                self.vectorizer = joblib.load(name_model + '.dic')
            except:
                file_h = open(name_model + '.dic', "rb")
                self.vectorizer = pickle.load(file_h)
                file_h.close()

        # print('Loading AM model')
        # with open(name_model + '.h5', 'rb') as f:
        #     self.am = pickle.load(f, encoding="bytes")
        # #self.am = joblib.load(name_model + '.h5')
        # file_h = open(name_model + '.dic', "rb")
        # self.vectorizer = pickle.load(file_h)
        # file_h.close()


def main():
    """Main method for the Test part, to call this script you must at least use the first 3 parameters, in addition check that the configuration file meets your models data"""
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="file with the gold standard sentences")
    parser.add_argument("out", help="file with the submission sentences")
    parser.add_argument("lang", help="target language [en|jp]")
    parser.add_argument("-list", "--list", action="store_true", help="ref and out files contains lists of parallel "
                                                                     "submission files to process. This speeds up the "
                                                                     "process since models are loaded only one time."
                                                                     "Submissions must be for the same language")
    parser.add_argument("-fm", "--fm", help="Do not calculate FM score", action="store_false")
    parser.add_argument("-am", "--am", help="Do not calculate AM score", action="store_false")
    parser.add_argument('-c', '--my-config', type=str, dest='MyConfigFilePath', help='config file path')
    parser.add_argument('-d', '--num_cores',type=int, dest='NofCores', help='Number of cores',default=1)  # this option can be set in a config file because it starts with '--'

    args = parser.parse_args()
    numcores = args.NofCores
    filepath = args.MyConfigFilePath   #path for the configuration File is input when executing script with -c argument

    if filepath == '' or filepath == None:
        print('You have not provided a configuration file')
        sys.exit()
    else:
        print('The current configuration file you are using is : ' + filepath)
        print('The number of Cores you setted up to use are: ' + str(numcores))


    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    print('The current models you are using is:' + str(cfg['test']['DATASET_DIR']))
    cs = calcScoresAMFM(lang=args.lang, am=True, fm=True, cfg=cfg)

    if args.list is True:
        list_ref = []
        with open(args.ref, 'r', 'utf-8') as f_in:
            for line in f_in.readlines():
                list_ref.append(line.strip())

        list_out = []
        with open(args.out, 'r', 'utf-8') as f_in:
            for line in f_in.readlines():
                list_out.append(line.strip())

        assert len(list_ref) == len(list_out), "******* ERROR: number of submissions and references are not the same " \
                                               "for files %s (%d) and %s (%d)" % (args.ref,
                                                                                  len(list_ref),
                                                                                  args.out,
                                                                                  len(list_out))

        for (ref, out) in zip(list_ref, list_out):
            processSubmissionNew(ref, out,cs, am=args.am, fm=args.fm,cfg=cfg)
    else:
        processSubmissionNew(args.ref, args.out, cs, am=args.am, fm=args.fm,cfg=cfg)


if __name__ == '__main__':
    main()