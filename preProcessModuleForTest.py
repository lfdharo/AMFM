#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'luisdhe' , 'vanmaren'
__doc__='this script objective is to do some preprocessing to the files that will be used for creating the AMFM models'

import subprocess
import os
import codecs
import random
import datetime
import multiprocessing
import argparse
from math import floor
import yaml
import sys
import string




#from spacy.tokenizer import Tokenizer







class PreProcessingClass():

    def __init__(self):
        global SCRIPTS_DIR
        global TRAIN_DATA_DIR



        self.sc = set(['-', "'","%"])
        self.to_remove = ''.join([c for c in string.punctuation if c not in self.sc])
        self.to_remove= self.to_remove +"“"+"”" #We had some extra characters to remove from texts
        self.table = dict((ord(char), u' ') for char in self.to_remove) #We replace the characters in the list(strg) with a blanc space = u' ' ( unicode blanc space)

        self.sc = set([',', '!', '?', '.'])
        self.to_separate = ''.join([c for c in string.punctuation if c not in self.sc])
        self.table_separate = dict((ord(char), u' ' + char + u' ') for char in self.to_separate) # We separate the characters in the list(strg) with a blanc space = u' ' ( unicode blanc space)


    # Basic method to perform basic Tokenization line by line

    def preProcessWork(self, sentence, lang):

        if len(sentence) == 0:  # To avoid empty lines
            return '_EMPTY_'

        # Remove some punctuation
        s = sentence.translate(self.table)
        if lang == 'my':  # we need to remove separation for | and -
            for k, v in self.table_separate.iteritems():
                if v == u' |' or v == u' -':
                    self.table_separate[k] = v.strip()
        # Replaces especial caracters in to " '@%' " = " ' @ % ' "
        s = s.translate(self.table_separate)

        # Tokenization by characters for most of Asian languages except for English, Indian, Korean and Myanmar

        if lang != 'en' and lang != 'in' and lang != 'ko'  and lang != 'my': #and lang != 'ta':and lang != 'hi'
            tokens = [' '.join([c for c in list(word.strip())]) for word in s.split()]

        else: #other langauges such as Chinese, Japanese, Hindi and Tamil
            tokens = s.split()
            if lang == 'my':  # We need to replace the | for -
                tokens = [x.replace('|', '-') for x in tokens]

        s = ' '.join(tokens)
        # s = ' '.join(tokens).lower()
        return s

    ######### FIRST STEP OF PREPROCESSING ############
    def tokenize(self, list_files_tokenize):
        print("***** Tokenizing files ******")
        t1 = datetime.datetime.now()
        print("Starting at " + str(t1))
        pool = multiprocessing.Pool(processes= self.cfg['preProcessModule']['NUM_CORES_MAX'])
        results = [pool.map(self.tokenize_wrapper, list_files_tokenize)]
        for res in results[0]:
            if res == "NO":
                print("ERROR: One of the tokenization files could not be created.")
                exit(-1)
        t2 = datetime.datetime.now()
        print("... Done tokenization in: %s " % (t2-t1))

    def tokenize_wrapper(self, args):
        return self.tokenizeFile(*args)

    def tokenizeFile(self, filename_in, lang, filename_out):

        if self.cfg['preProcessModule']['Tokenize'] == True: #si queremos ejecutar la tokenizacion
            with open(filename_in, 'r') as f_in, open(filename_out, 'w',encoding='utf-8') as f_out:
                for line in f_in: #f_in.readlines():
                     linetokenized = self.preProcessWork(line,lang)
                     f_out.write(linetokenized+'\n')

        else:
            cmd = self.cfg['preProcessModule']['Tokenize_Script']  #insert your own line for tokenizing files
        #instead of using the cmd script you can use any other script for tokenizing the files


        if not os.path.exists(filename_out) or os.path.getsize(filename_out) == 0:
            print('ERROR: tokenizing file ' + filename_in)
            return "NO"
        print("... Done " + filename_in + " saving in " + filename_out)
        return "OK"

    ##################################################


    ######### THIRD STEP OF PREPROCESSING ############
    def clean(self, list_files_clean):
        print("***** cleaning files ******")
        t1 = datetime.datetime.now()
        print("Starting at " + str(t1))
        # pool = multiprocessing.Pool(processes=NUM_CORES_MAX)
        pool = multiprocessing.Pool(processes=self.cfg['preProcessModule']['NUM_CORES_MAX'])
        results = [pool.map(self.clean_wrapper, list_files_clean)]
        for res in results[0]:
            if res == "NO":
                print("ERROR: One of the cleaning files could not be created.")
                exit(-1)
        t2 = datetime.datetime.now()
        print("... Done cleaning in: %s " % (t2 - t1))

    def clean_wrapper(self, args):
        return self.cleanFile(*args)

    def cleanFile(self, file, src, tgt):
        filename_in_src = file + '.lower.' + src
        filename_in_tgt = file + '.lower.' + tgt

        file_out_src_clean = file + '.clean.' + src
        file_out_tgt_clean= file + '.clean.' + tgt

        if src in self.lang_char_tokenization:
            MIN_SENTENCE_SRC_LENGTH = self.cfg['preProcessModule']['MIN_SENTENCE_LENGTH_CHARS']
            MAX_SENTENCE_SRC_LENGTH = self.cfg['preProcessModule']['MAX_SENTENCE_LENGTH_CHARS']
        else:
            MIN_SENTENCE_SRC_LENGTH = self.cfg['preProcessModule']['MIN_SENTENCE_LENGTH']
            MAX_SENTENCE_SRC_LENGTH = self.cfg['preProcessModule']['MAX_SENTENCE_LENGTH']

        if tgt in self.lang_char_tokenization:
            MIN_SENTENCE_TGT_LENGTH = self.cfg['preProcessModule']['MIN_SENTENCE_LENGTH_CHARS']
            MAX_SENTENCE_TGT_LENGTH = self.cfg['preProcessModule']['MAX_SENTENCE_LENGTH_CHARS']
        else:
            MIN_SENTENCE_TGT_LENGTH = self.cfg['preProcessModule']['MIN_SENTENCE_LENGTH']
            MAX_SENTENCE_TGT_LENGTH = self.cfg['preProcessModule']['MAX_SENTENCE_LENGTH']

        with open(filename_in_src, 'r') as r1, open(filename_in_tgt, 'r') as r2, \
                open(file_out_src_clean,'w',encoding='utf-8') as w1, open(file_out_tgt_clean, 'w',encoding='utf-8') as w2:
            for line_src, line_tgt in zip(r1, r2):
                len_line_src = len(line_src.split())
                len_line_tgt = len(line_tgt.split())
                if (len_line_src > MAX_SENTENCE_SRC_LENGTH or len_line_src < MIN_SENTENCE_SRC_LENGTH or
                    len_line_tgt > MAX_SENTENCE_TGT_LENGTH or len_line_src < MIN_SENTENCE_TGT_LENGTH):
                    continue
                else:
                    w1.write(line_src)
                    w2.write(line_tgt)

        if not os.path.exists(file_out_src_clean) or os.path.getsize(file_out_src_clean) == 0:
            print('ERROR: cleaning file ' + file_out_src_clean)
            return "NO"
        elif not os.path.exists(file_out_tgt_clean) or os.path.getsize(file_out_tgt_clean) == 0:
            print('ERROR: cleaning file ' + file_out_tgt_clean)
            return "NO"

        print("... Done " + file + " saving in " + file_out_src_clean + ' and ' + file_out_tgt_clean)
        return "OK"
    ##################################################

    ########## SECOND STEP OF PREPROCESSING ############
    def preProcessing(self, list_files_lowercase):
        print("***** Lowercasing files ******")
        t1 = datetime.datetime.now()
        print("Starting at " + str(t1))
        # pool = multiprocessing.Pool(processes=NUM_CORES_MAX)
        pool = multiprocessing.Pool(processes=self.cfg['preProcessModule']['NUM_CORES_MAX'])
        results = [pool.map(self.preproc_wrapper, list_files_lowercase)]
        for res in results[0]:
            if res == "NO":
                print("ERROR: One of the lowercase files could not be created.")
                exit(-1)
        t2 = datetime.datetime.now()
        print("... Done lowercasing in: %s " % (t2-t1))

    def preproc_wrapper(self, args):
        return self.preprocFile(*args)

    def preprocFile(self, filename_in, lang, filename_out):

        #metodo para pasar a lower casing
        try:
            with open(filename_in, 'r') as f_in, open(filename_out, 'w',encoding='utf-8') as f_out:
                for line in f_in.readlines():
                    f_out.write(line.lower())
            return "OK"
        except:
            print("There was an error when processing file %s" % filename_in)
            return "NO"

    ##################################################




def main(): ###Aqui deberimos buscar al forma de meter las variables que se usan en general en los metodos, ademas estas variables cuando lsa funciones son invcoadas desde el runall deben de pasar los parametros??

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--my-config', type=str, dest='MyConfigFilePath', required=False,help='config file path')
    parser.add_argument('-overwrite', help='Overwrite all the files.', action='store_true')
    parser.add_argument('-d', '--num_cores', dest='NofCores', help='Number of cores')  # this option can be set in a config file because it starts with '--'


    args = parser.parse_args()

    filepath = args.MyConfigFilePath

    if filepath == '' or filepath == None:
        print('You have not provided a configuration file')
        sys.exit()

    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    submissionsPerLanguagePerYear = cfg['preProcessModule']['submissionsPerLanguagePerYear']
    FilesPerLanguageForLM = cfg['preProcessModule']['filesPerLanguageForLM']
    FilesPerLanguage = cfg['preProcessModule']['filesPerLanguage']

    root_dir = cfg['directories']['INPUT'] # con esta sentencia estamos partiedno el origen desde el Train para bajo

    submissions_dir = cfg['preProcessModule']['submissions_dir']
    train_data_dir = cfg['preProcessModule']['train_data_dir']
    scripts_dir = cfg['preProcessModule']['scripts_dir']

    overwrite_all = args.overwrite


    clPrep = PreProcessingClass(submissionsPerLanguagePerYear=submissionsPerLanguagePerYear,
                                filesPerLanguage=FilesPerLanguage,
                                filesPerLanguageForLM=FilesPerLanguageForLM,
                                train_data_dir=train_data_dir,
                                scripts_dir=scripts_dir,
                                submissions_dir=submissions_dir,
                                cfg=cfg)

    clPrep.preprocess_files(bDoAll=overwrite_all)
    clPrep.createTrainingFiles(FilesPerLanguage, overwrite_all=overwrite_all)


if __name__ == '__main__':
    main()
