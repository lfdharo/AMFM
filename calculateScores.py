# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = 'luisdhe'


import numpy as np
#import cPickle as pickle
import pickle as pickle
import os
import signal
import re
import subprocess, shlex
import codecs
import general as gen
from vector_space import VectorSpace
import argparse
import datetime
import multiprocessing
import yaml

from recsys_evaluation.ranking import AveragePrecision
from recsys_evaluation.ranking import MeanAveragePrecision
from recsys_evaluation.prediction import MAE

with open('/home/enrique/Escritorio/TFG_Pendrive/AMFM/Settings.yaml', 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

root_dir = os.path.dirname(os.path.realpath(__file__))
scores_dir = root_dir + '/scores/'
models_dir = root_dir + '/models/'
dir_lm_out = root_dir + '/lms/'

type_vectorizer = 'counts'
dir_svd_mono = root_dir + '/svd_mono_' + type_vectorizer
dir_svd_cross = root_dir + '/svd_cross_' + type_vectorizer

MIN_COUNTS = cfg['calculatescores']['MIN_COUNTS']  # Number of times a word must occur to be included in the SVM model
MIN_NGRAM_ORDER = cfg['calculatescores']['MIN_NGRAM_ORDER']
MAX_NGRAM_ORDER = cfg['calculatescores']['MAX_NGRAM_ORDER']
NUM_MAX_CORES = cfg['calculatescores']['NUM_MAX_CORES']
MIN_LOG_PROB = cfg['calculatescores']['MIN_LOG_PROB']
STARTING_VALUE_FEATURES = cfg['calculatescores']['STARTING_VALUE_FEATURES']
NFOLDS = cfg['calculatescores']['NFOLDS']  # Number of NFOLDS cross-training sets we are creating for the AM experiments

overwrite_all = False

filesPerLanguageForLM = {
    'en': [
        'ASPEC/ASPEC-JE/train/train',
    ],
    'jp': [
        'ASPEC/ASPEC-JE/train/train',
    ],
}

filesPerLanguage = {
    'en-jp': [
        'ASPEC/ASPEC-JE/train/train-1',
        'ASPEC/ASPEC-JE/train/train-2',
        'ASPEC/ASPEC-JE/train/train-3',
    ],
    'jp-en': [
        'ASPEC/ASPEC-JE/train/train-1',
        'ASPEC/ASPEC-JE/train/train-2',
        'ASPEC/ASPEC-JE/train/train-3',
    ],
}

submissionsPerLanguagePerYear = {
    'en-jp':{
        '2015':{
           'source': 'ASPEC/ASPEC-JE/devtest/devtest.en',
           'reference': 'ASPEC/ASPEC-JE/devtest/devtest.jp.ref',
           'test_id': 'devtest_en-jp',
           'submissions': [
               'ASPEC/ASPEC-JE/devtest/devtest.jp.google',
           ],
        },
    },
    'jp-en':{
        '2015':{
           'source': 'ASPEC/ASPEC-JE/devtest/devtest.jp',
           'reference': 'ASPEC/ASPEC-JE/devtest/devtest.en.ref',
           'test_id': 'devtest_jp-en',
           'submissions': [
               'ASPEC/ASPEC-JE/devtest/devtest.en.google',
           ],
        },
    },
}

monoSizesSVD = dict()
crossSizesSVD = dict()
aSizesTrain = [10000]
monoSizesSVD[10000] = [100, 200, 300, 500, 1000, 2000, 2500, 8000]  # The last one must be always the maximum value of the model
crossSizesSVD[10000] = [50, 100, 150, 250]

def am_mono_scores_wrapper(args):
    return calculateMonolingualScores(*args)


def am_cross_scores_wrapper(args):
    return calculateCrosslingualScores(*args)


def lm_wrapper(args):
    return calculateLMMetricsPerTaskParallel(*args)


def am_scores_wrapper(args):
    return calculateMonolingualScores(*args)


def createAndcheckOutputDirs():
    print("***** Checking input and creating output directories *****")
    if not os.path.exists(dir_svd_mono):
        print("ERROR: the svd dir for the monolingual matrices does not exist")
        exit(-1)

    if not os.path.exists(dir_svd_cross):
        print("ERROR: the svd dir for the crosslingual matrices does not exist")
        exit(-1)

    if not os.path.exists(scores_dir):
        print("...creating " + scores_dir)
        os.makedirs(scores_dir)

    print("... Done")


def calculateLMMetricsPerTaskParallel(year, tgt, filename, ngram_order):
    print("Calculating LM Metrics for year: " + year + " lang:" + tgt + "filename: " + filename + " and ngram:" +
          str(ngram_order) + " ******")
    t1 = datetime.datetime.now()
    lm_scores_systems = []

    basenamefile = os.path.basename(filename)
    # Extracts from the original submissions only those that were considered in the judgement files.
    # This section extracts the FM metric
    ppl_out_dir = models_dir + '/' + year + '/' + tgt + '/'
    if not os.path.exists(ppl_out_dir):
        print("...creating " + ppl_out_dir)
        os.makedirs(ppl_out_dir)

    # Check that the text file for the target language exists
    if not os.path.exists(root_dir + '/' + filename + '.lower'):
        print("******* ERROR: file " + root_dir + filename + '.lower' + ' does not exists.')
        exit(-1)

    # Check the PPL for the test file.
    ppl_file_path_test = ppl_out_dir + basenamefile + '.' + str(ngram_order) + '.ppl_in'

    if not os.path.isfile(ppl_file_path_test):
        lm_path = dir_lm_out + tgt + '/' + tgt + '.' + str(ngram_order) + '.lm'
        #cmd_in = 'ngram -debug 2 -ppl ' + root_dir + filename + '.lower -unk -lm ' + lm_path + ' -order ' + str(ngram_order) + ' 2>&1 | tee ' + ppl_file_path_test
        #cmd_in = 'ngram -ppl ' + root_dir + filename + '.lower -use-server 10000@localhost -cache-served-ngrams -debug 2 > ' + ppl_file_path_test
        cmd_in = 'ngram -ppl ' + root_dir + '/' + filename + \
                 '.lower -use-server 1000' + str(ngram_order) +'@localhost -cache-served-ngrams -debug 2 -order ' + \
                 str(ngram_order) + ' > ' + ppl_file_path_test
        r = os.system(cmd_in)

    num_sentences = 0
    h_in = open(ppl_file_path_test)
    num_words = 0
    sum_log_prob = 0
    for line in h_in.readlines():
        # Process each line
        if line.startswith("file "): # Ends when the last lines are printed out
            break
        m = re.search("p\(.*\)\s+=\s+(.*)\s+\[(.*)\]", line.strip())
        log_prob = re.search(", logprob= (.*) ppl=", line)
        if m is not None:
            num_words += 1
            sum_log_prob += float(m.group(2))
        elif log_prob is not None:
            lm_scores_systems.append(1.0 - (sum_log_prob/float(num_words*MIN_LOG_PROB)))
            sum_log_prob = 0
            num_sentences += 1

    sum_probs = 0
    for value in lm_scores_systems:
        sum_probs += value

    if num_sentences != 0:
        print("Year: " + year + " lang:" + tgt + " filename: " + filename + " ngram:" + str(ngram_order)
              + " avg: " + str(sum_probs/num_sentences))

    t2 = datetime.datetime.now()
    print("Execution time: %s" % (t2-t1))
    return lm_scores_systems


def calculateMonolingualScores(size, src, tgt, year, filename, reference_file, mono_model_path, bAverageFiles=False):
    print('Calculating mono AM scores for: size:' + str(size) + ' year:' + str(year) +
          ' lang_pair:(' + src + '-' + tgt + ') file:' + filename)
    t1 = datetime.datetime.now()

    scores_allsvdsizes= {}
    scores_allfiles = {}
    test_data = codecs.open(root_dir + '/' + filename + '.lower', 'r', 'utf-8')
    test_sentences = test_data.readlines()
    test_data.close()

    ref_data = codecs.open(root_dir + '/' + reference_file + '.lower', 'r', 'utf-8')
    ref_sentences = ref_data.readlines()
    ref_data.close()

    assert len(test_sentences) == len(ref_sentences), "******* ERROR: sentence lengths are not the same for files %s " \
                                                      "(%d) and %s (%d)" % (root_dir + '/' + filename + '.lower',
                                                                            len(test_sentences),
                                                                            root_dir + reference_file,
                                                                            len(ref_sentences))

    max_size = monoSizesSVD[size][-1]
    for svd_size in monoSizesSVD[size]:
        scores_allfiles[svd_size] = np.empty([len(filesPerLanguage[src + '-' + tgt]), len(ref_sentences)])

    ap = AveragePrecision()

    for (id_file, svd_file) in enumerate(filesPerLanguage[src + '-' + tgt]):
        for nF in range(NFOLDS):
            # This is the full matrix without any reduction
            svd_full_matrix = dir_svd_mono + '/' + svd_file + '.' + tgt + '.' + str(size) + '.' + str(max_size) + \
                              '.' + str(nF)
            if os.path.isfile(svd_full_matrix + '.h5') and os.path.isfile(svd_full_matrix + '.dic'):
                # Open the full SVD and calculate the score
                vs = VectorSpace()
                vs.load(svd_full_matrix)
                for svd_size in monoSizesSVD[size]:
                    print("SVD SIZE: " + str(svd_size) + " and FOLD: " + str(nF))
                    if svd_size not in scores_allsvdsizes:
                        # We need to save all the possible cosine distances
                        scores_allsvdsizes[svd_size] = np.empty([NFOLDS, len(ref_sentences)])
                    score = vs.search(tgt, ref_sentences, test_sentences, svd_size)
                    scores_allsvdsizes[svd_size][nF] = np.diagonal(score)

                    DATA_PRED = []
                    # Calculates MEAN AVERAGE ERROR
                    for i in range(len(ref_sentences)):
                        DATA_PRED.append((1.0, min(1.0, score[i][i])))
                    mae = MAE(DATA_PRED)
                    score_mae = mae.compute()

                    Map = MeanAveragePrecision()
                    P=1  # Calculates precision@1
                    for i in range(len(ref_sentences)):
                        Q = np.argsort(-score[i])
                        Map.load([i], list(Q[0:P]))
                    score_map = Map.compute()
                    score = np.average(np.diagonal(score))
                    print("SVD: " + svd_file + " SIZE: " + str(size) + " SRC: " + src + " TGT: " + tgt + " NFOLD:" +
                          str(nF) + " MAE: " + str(score_mae) + " MAP: " + str(score_map) + " AVG: " + str(score))
            else:
                print('******* ERROR: files: ' + svd_full_matrix + '.h5 or ' + svd_full_matrix + '.dic do not exists.')
                return
                # exit(-1)

        # Take the average of the different folds for a given SVD
        for svd_size in monoSizesSVD[size]:
            scores_allfiles[svd_size][id_file] = np.average(scores_allsvdsizes[svd_size], axis=0)

    if bAverageFiles is True:
        # Take the average of the different SVD matrices
        for svd_size in monoSizesSVD[size]:
            scores_allfiles[svd_size] = np.average(scores_allfiles[svd_size], axis=0)

    print("... Done")
    if bAverageFiles is True:
        outFile = open(mono_model_path + '.' + str(size) + '.am_mono_model_avg', "wb")
    else:
        outFile = open(mono_model_path + '.' + str(size) + '.am_mono_model', "wb")	
    pickle.dump(scores_allfiles, outFile, -1)
    outFile.close()

    t2 = datetime.datetime.now()
    print("... Done size " + str(size))
    print("Execution time: %s" % (t2-t1))
    return scores_allfiles


def calculateCrosslingualScores(size, src, tgt, year, filename, source_file, cross_model_path, bAverageFiles=False):
    print('Calculating cross AM scores for: size:' + str(size) + ' year:' + str(year) +
          ' lang_pair:(' + src + '-' + tgt + ') file:' + filename)
    t1 = datetime.datetime.now()

    test_data = codecs.open(root_dir + '/' + filename + '.lower', 'r', 'utf-8')
    test_sentences = test_data.readlines()
    test_data.close()

    src_data = codecs.open(root_dir + '/' + source_file + '.lower', 'r', 'utf-8')
    src_sentences = src_data.readlines()
    src_data.close()

    scores_allsvdsizes= {}
    scores_allfiles = {}
    lang = tgt
    if tgt != 'en': # Since we only trained half of the matrices we just change the name to get the right matrix
        lang = src
        test_sentences, src_sentences = src_sentences, test_sentences  # swap the variables

    assert len(test_sentences) == len(src_sentences), "******* ERROR: sentence lengths are not the same for " \
                                                      "files %s (%d) and %s (%d)" % (root_dir + '/' + filename + '.lower',
                                                                                     len(test_sentences), root_dir +
                                                                                     source_file, len(src_sentences))

    max_size = crossSizesSVD[size][-1]
    for svd_size in crossSizesSVD[size]:
        scores_allfiles[svd_size] = np.empty([len(filesPerLanguage[src + '-' + tgt]), len(test_sentences)])

    for (id_file, svd_file) in enumerate(filesPerLanguage[src + '-' + tgt]):
        for nF in range(NFOLDS):
            # This is the full matrix without any reduction
            svd_full_matrix = dir_svd_cross + '/' + svd_file + '.' + lang + '.' + str(size) + '.' + str(max_size) + \
                              '.' + str(nF)
            if os.path.isfile(svd_full_matrix + '.h5') and os.path.isfile(svd_full_matrix + '.dic'):
                # Open the full SVD and calculate the score
                vs = VectorSpace()
                vs.load(svd_full_matrix)
                for svd_size in crossSizesSVD[size]:
                    print("SVD SIZE: " + str(svd_size) + " FOLD: " + str(nF))
                    if svd_size not in scores_allsvdsizes:
                        scores_allsvdsizes[svd_size] = np.empty([NFOLDS, len(src_sentences)])

                    score = vs.search(src_sentences, test_sentences, svd_size, True)
                    scores_allsvdsizes[svd_size][nF] = np.diagonal(score)
                    DATA_PRED = []
                    for i in range(len(ref_sentences)):
                        DATA_PRED.append((1.0, min(1.0, score[i][i])))
                    mae = MAE(DATA_PRED)
                    score_mae = mae.compute()

                    Map = MeanAveragePrecision()
                    for i in range(len(ref_sentences)):
                        Q = np.argsort(-score[i])
                        Map.load([i], Q[0])
                    score_map = Map.compute()

                    score = np.average(score)
                    print("SVD: " + svd_file + " SIZE: " + str(size) + " SRC: " + src + " TGT: " + tgt + " NFOLD:" +
                          str(nF) + " MAE: " + str(score_mae) + " MAP: " + str(score_map) + " AVG: " + str(score))
            else:
                print('******* ERROR: files: ' + svd_full_matrix + '.h5 or ' + svd_full_matrix + '.dic do not exists.')
                exit(-1)

        # Take the average of the different SVD files created
        for svd_size in crossSizesSVD[size]:
            scores_allfiles[svd_size][id_file] = np.average(scores_allsvdsizes[svd_size], axis=0)

    if bAverageFiles is True:
        # Take the average of the different SVD matrices
        for svd_size in crossSizesSVD[size]:
            scores_allfiles[svd_size] = np.average(scores_allfiles[svd_size], axis=0)

    print("... Done")
    if bAverageFiles is True:
        outFile = open(cross_model_path + '.' + str(size) + '.am_cross_model_avg', "wb")
    else:
        outFile = open(cross_model_path + '.' + str(size) + '.am_cross_model', "wb")
    pickle.dump(scores_allfiles, outFile, -1)
    outFile.close()

    t2 = datetime.datetime.now()
    print("... Done size " + str(size))
    print("Execution time: %s" % (t2-t1))
    return scores_allfiles


def lm_wrapper(args):
   return calculateLMMetricsPerTaskParallel(*args)


def calculateLMMetrics(tgt, lang_pair, year, submissions):
    # This section will calculate the LM metrics for the submitted systems. The system saves and retrieves an
    # array containing this information for each n-gram order
    print("Calculate PPL Metrics")
    pool = multiprocessing.Pool(processes=NUM_MAX_CORES)
    lm_scores_per_sentence_allorders = {}

    for ngram_order in range(MIN_NGRAM_ORDER, MAX_NGRAM_ORDER+1, 1):
        # Run the server
        lm_path = dir_lm_out + '/' + tgt + '/' + 'train.' + tgt + '.' + str(ngram_order) + '.lm'
        cmd_in = 'ngram -unk -lm ' + lm_path + ' -order ' + str(ngram_order) + ' -server-port 1000' +\
                 str(ngram_order) + ' 2>&1 '
        p_server = subprocess.Popen(cmd_in, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    preexec_fn=os.setsid)
        bFlag = True
        while (bFlag):
            line = p_server.stdout.readline()
            if line.strip() == 'starting prob server on port 1000' + str(ngram_order):
                bFlag = False
                break
            elif line.find('No such file or directory') != -1:
                print('ERROR: ' + line)
                exit(-1)

        tmpTask = []
        for filename in submissions:
            filenamebase = os.path.basename(filename)
            print("Order: " + str(ngram_order) + ' file:' + filenamebase)
            tmpTask.append((year, tgt, filename, ngram_order))

        values = pool.map(lm_wrapper, tmpTask)
        for (id, filename) in enumerate(submissions):
            filenamebase = os.path.basename(filename)
            if filenamebase not in lm_scores_per_sentence_allorders:
                lm_scores_per_sentence_allorders[filenamebase] = {}
            lm_scores_per_sentence_allorders[filenamebase][ngram_order] = values[id]

        os.killpg(p_server.pid, signal.SIGTERM)  # Send the signal to all the process groups

    for (id, filename) in enumerate(submissions):
        filenamebase = os.path.basename(filename)
        lm_model_path = models_dir + '/' + year + '/' + lang_pair + '/' + filenamebase + '.fm_model'
        if not os.path.exists(os.path.dirname(lm_model_path)):
            os.makedirs(os.path.dirname(lm_model_path))

        outFile = open(lm_model_path, "wb")
        pickle.dump(lm_scores_per_sentence_allorders[filenamebase], outFile, -1)
        outFile.close()

    print("... Done")


def fntProcessSubmissions(submissionsPerLanguagePerYear, overwrite_all=False, bAverageFiles=False, fpl=None):
    cross = False
    if fpl is not None:
        global filesPerLanguage
        filesPerLanguage = fpl

    for (lang_pair, years) in submissionsPerLanguagePerYear.items():
        print("Processing pair:" + lang_pair)
        aPrefix = lang_pair.split("-")
        src = aPrefix[0]
        tgt = aPrefix[1]

        for (year, info) in submissionsPerLanguagePerYear[lang_pair].items():
            print("Processing year:" + year)
            model_path = models_dir + year + '/' + lang_pair
            if not os.path.exists(model_path):
                print("...creating " + model_path)
                os.makedirs(model_path)
            reference_file = info['reference']
            source_file = info['source']
            test_id = info['test_id']

            # Calculate all the PPL for a given target domain and year
            # ToDo: Check that output files do not exist to avoid entering here
            calculateLMMetrics(tgt, lang_pair, year, submissionsPerLanguagePerYear[lang_pair][year]['submissions'])

            for filename in info['submissions']:
                print('Processing file:' + filename)
                filenamebase = os.path.basename(filename)
                lm_model_path = model_path + '/' + filenamebase + '.fm_model'
                print('FM scores for year:' + year + ' lang_pair:' + lang_pair + ' file:' + filenamebase)
                if os.path.isfile(lm_model_path) and overwrite_all is False:  # The model exists then load it
                    fm_scores_sent = gen.load_lm_model(lm_model_path)
                else:
                    print('****** ERROR: LM: ' + lm_model_path + ' could not be loaded')
                    exit(-1)

                exTAvg = ''
                if bAverageFiles is True:
                    exTAvg = '_avg'

                mono_model_path = model_path + '/' + filenamebase
                am_mono_scores_sent = {}
                print('AM Mono scores for year:' + year + ' lang_pair:' + lang_pair + ' file:' + filenamebase)

                print('Calculating am mono scores')
                tmpParams = []
                for train_size in aSizesTrain:
                    nameModel = mono_model_path + '.' + str(train_size) + '.am_mono_model' + exTAvg
                    if overwrite_all is True or not os.path.exists(nameModel):
                        tmpParams.append((train_size, src, tgt, year, filename, reference_file, mono_model_path,
                                          bAverageFiles))
                if len(tmpParams) > 0:
                    pool = multiprocessing.Pool(processes=NUM_MAX_CORES)
                    pool.map(am_mono_scores_wrapper, tmpParams)

                for train_size in aSizesTrain:
                    nameModel = mono_model_path + '.' + str(train_size) + '.am_mono_model' + exTAvg
                    if os.path.exists(nameModel):
                        am_mono_scores_sent[train_size] = gen.load_am_model(nameModel)
                    else:
                        print('******* ERROR: File:' + nameModel + ' could not be loaded.')
                        exit(-1)

                if cross is True:
                    cross_model_path = model_path + '/' + filenamebase
                    am_cross_scores_sent = {}
                    print('AM Cross scores for year:' + year + ' lang_pair:' + lang_pair + ' file:' + filenamebase)
                    print('Calculating am cross scores')
                    tmpParams = []
                    for train_size in aSizesTrain:
                        nameModel = cross_model_path + '.' + str(train_size) + '.am_cross_model' + exTAvg
                        if overwrite_all is True or not os.path.exists(nameModel):
                            tmpParams.append((train_size, src, tgt, year, filename, source_file, cross_model_path,
                                              bAverageFiles))
                    if len(tmpParams) > 0:
                        pool = multiprocessing.Pool(processes=NUM_MAX_CORES)
                        pool.map(am_cross_scores_wrapper, tmpParams)


                    for train_size in aSizesTrain:
                        nameModel = cross_model_path + '.' + str(train_size) + '.am_cross_model' + exTAvg
                        if os.path.isfile(nameModel):
                            am_cross_scores_sent[train_size] = gen.load_am_model(nameModel)
                        else:
                            print('******* ERROR: File:' + nameModel + ' model could not loaded.')
                            exit(-1)

                file_out = scores_dir + year + '/' + lang_pair + '/' + filenamebase + '.scores'
                dir_out = os.path.dirname(file_out)
                if not os.path.exists(dir_out):
                    os.makedirs(dir_out, 755)
                h_file_out = codecs.open(file_out, 'w', 'utf-8')
                print("Writing score file: " + file_out)
                for i in range(len(fm_scores_sent[MIN_NGRAM_ORDER])):
                    # <METRIC NAME>   <LANG-PAIR>   <TEST SET>   <SYSTEM>   <SEGMENT NUMBER>   <SEGMENT SCORE>
                    #sent = "am_xam_fm\t" + lang_pair + "\t" + test_id + "\t" + filenamebase + "\t" + str(i+1) + "\t"
                    num_feature = STARTING_VALUE_FEATURES
                    sent = ''
                    for ngram_order in range(MIN_NGRAM_ORDER, MAX_NGRAM_ORDER+1, 1):
                        sent += str(num_feature) + ':' + str(fm_scores_sent[ngram_order][i]) + " "
                        num_feature += 1

                    if bAverageFiles is False:
                        for train_size in aSizesTrain:
                            for key in sorted(am_mono_scores_sent[train_size]):
                                for valuesPerFile in am_mono_scores_sent[train_size][key]:
                                    sent += str(num_feature) + ':' + str(valuesPerFile[i]) + " "
                                    num_feature += 1
                            if cross is True:
                                for key in sorted(am_cross_scores_sent[train_size]):
                                    for valuesPerFile in am_cross_scores_sent[train_size][key]:
                                        sent += str(num_feature) + ':' + str(valuesPerFile[i]) + " "
                                        num_feature += 1
                    else:
                        for train_size in aSizesTrain:
                            for key in sorted(am_mono_scores_sent[train_size]):
                                sent += str(num_feature) + ':' + str(am_mono_scores_sent[train_size][key][i]) + " "
                                num_feature += 1
                            if cross is True:
                                for key in sorted(am_cross_scores_sent[train_size]):
                                    sent += str(num_feature) + ':' + str(am_cross_scores_sent[train_size][key][i]) + " "
                                    num_feature += 1

                    sent += "\n"
                    h_file_out.write(sent)

                h_file_out.close()
    print('DONE ALL')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-overwrite', help='Overwrite all the files.', action='store_true')
    parser.add_argument('-average', help='Average the results from different SVD files.', action='store_true')

    args = parser.parse_args()
    overwrite_all = args.overwrite
    bAverageFiles = args.average

    # Check input directories and Creates all the directories required to save output information
    createAndcheckOutputDirs()
    fntProcessSubmissions(submissionsPerLanguagePerYear, overwrite_all, bAverageFiles)
    exit(-1)

if __name__ == '__main__':
    # svd_full_matrix='/home/luisdhe/Work/WMT15/svd_mono_tfidf/training-parallel-nc-v10/news-commentary-v10.fr-en.en.1000.0.mc0'
    # vs = VectorSpace()
    # vs.load(svd_full_matrix)
    main()


