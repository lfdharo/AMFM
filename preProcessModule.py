__author__ = 'luisdhe'

import subprocess
import os
import codecs
import random
import datetime
import multiprocessing
import argparse
from math import floor
import yaml

# with open('/home/enrique/Escritorio/TFG_Pendrive/AMFM/Settings.yaml', 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

SCRIPTS_DIR = None
TRAIN_DATA_DIR = None

# SCRIPTS_DIR = cfg['preProcessModule']['SCRIPTS_DIR']
# TRAIN_DATA_DIR = cfg['preProcessModule']['TRAIN_DATA_DIR']

filesPerLanguage = {
    'en-jp': [
        'ASPEC/ASPEC-JE/train/train-1',
        'ASPEC/ASPEC-JE/train/train-2',
        'ASPEC/ASPEC-JE/train/train-3',
    ],
}

lang_char_tokenization = ['jp', 'cn', 'ko']

#
# def tokenize_wrapper(args):
#     return tokenizeFile(*args)
#
#
# def tokenizeFile(filename_in, lang, filename_out):
#     cmd = 'perl ' + SCRIPTS_DIR + 'perl/tokenizer.perl -time -no-escape -threads 3 -l ' + lang + ' < ' + filename_in \
#           + ' > ' + filename_out
#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
#     print('----' + filename_in + ' : ' + p.decode())
#     # for line in p.stdout.readlines():
#     # for line in p.readlines():
#     #    print('----' + filename_in + ' : ' + line)
#     # retval = p.wait()
#     if not os.path.exists(filename_out) or os.path.getsize(filename_out) == 0:
#         print('ERROR: tokenizing file ' + filename_in)
#         return "NO"
#     print("... Done " + filename_in + " saving in " + filename_out)
#     return "OK"
#
#
# def clean_wrapper(args):
#     return cleanFile(*args)
#
#
# def cleanFile(file, src, tgt):
#     filename_out_src = file + '.token.' + src
#     filename_out_tgt = file + '.token.' + tgt
#
#     if src in lang_char_tokenization or tgt in lang_char_tokenization:
#         cmd = 'perl ' + SCRIPTS_DIR + 'perl/clean-corpus-n.perl -lc ' + file + '.token ' + src + ' ' + tgt + ' ' + file\
#               + '.clean ' + str(MIN_SENTENCE_LENGTH_CHARS) + ' ' + str(MAX_SENTENCE_LENGTH_CHARS)
#     else:
#         cmd = 'perl ' + SCRIPTS_DIR + 'perl/clean-corpus-n.perl -lc ' + file + '.token ' + src + ' ' + tgt + ' ' + file\
#               + '.clean ' + str(MIN_SENTENCE_LENGTH) + ' ' + str(MAX_SENTENCE_LENGTH)
#
#     # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     # for line in p.stdout.readlines():
#     #     print('----' + file + ' > ' + line)
#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
#     print('----' + file + ' > ' + p.decode())
#     # retval = p.wait()
#     if not os.path.exists(filename_out_src) or os.path.getsize(filename_out_src) == 0:
#         print('ERROR: tokenizing file ' + filename_out_src)
#         return "NO"
#     elif not os.path.exists(filename_out_tgt) or os.path.getsize(filename_out_tgt) == 0:
#         print('ERROR: tokenizing file ' + filename_out_tgt)
#         return "NO"
#
#     print("... Done " + file + " saving in " + filename_out_src + ' and ' + filename_out_tgt)
#     return "OK"
#
#
# def preproc_wrapper(args):
#     return preprocFile(*args)
#
#
# def preprocFile(filename_in, lang, filename_out):
#     # cmd = 'perl ' + SCRIPTS_DIR + 'perl/replace-unicode-punctuation.perl ' + ' < ' + filename_in
#     # cmd += ' | perl ' + SCRIPTS_DIR + 'perl/remove-non-printing-char.perl - '
#     # cmd += ' | perl ' + SCRIPTS_DIR + 'perl/normalize-punctuation.perl -l ' + lang + ' - '
#     # cmd += ' | perl ' + SCRIPTS_DIR + 'perl/lowercase.perl ' + ' - > ' + filename_out
#     # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#     # for line in p.stdout.readlines():
#     #     print('---- ' + filename_in + ' : ' + line)
#     # retval = p.wait()
#     # if not os.path.exists(filename_out) or os.path.getsize(filename_out) == 0:
#     #     print('ERROR: tokenizing file ' + filename_in)
#     #     return "NO"
#     # print("... Done " + filename_in + " saving in " + filename_out)
#     try:
#         with open(filename_in, 'r') as f_in, open(filename_out, 'w') as f_out:
#             for line in f_in.readlines():
#                 f_out.write(line.lower())
#         return "OK"
#     except:
#         print("There was an error when processing file %s" % filename_in)
#         return "NO"
#
#
# def createtrainfile_wrapper(args):
#     return createTrainingFilesParallel(*args)
#
#
# def createTrainingFilesParallel(file, src, tgt, make_new=False):
#     print("***** Creating training file for " + file + ' src: ' + src + ' and tgt: ' + tgt + " ******")
#
#     train_file_path_src = file + '.lower.' + src
#     train_file_path_tgt = file + '.lower.' + tgt
#
#     if os.path.isfile(train_file_path_src) and os.path.isfile(train_file_path_tgt):
#         # if (make_new is True) :
#         # First step is to read the src and load unique lines
#         dict_src = {}
#         useful_lines = []
#         iNumLine = 0
#         #with codecs.open(train_file_path_src, 'r', 'utf-8', errors='ignore') as f: buffering=None, encoding=None, errors=None, newline=None, closefd=True
#         with open(train_file_path_src, 'r', encoding='utf-8', errors='ignore') as f:
#             for src_sent in f:
#                # src_sent= unicode(src_sent,errors='ignore')
#                 if src_sent not in dict_src:
#                     dict_src[src_sent] = 1
#                     useful_lines.append(iNumLine)  # Save the line number
#                 iNumLine += 1
#         # len_src = len(useful_lines)
#         del dict_src  # Save memory
#
#         dict_tgt = {}
#         iNumLine = 0
#         pointerSrc = 0
#         final_useful_lines = []
#         # with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
#         with open(train_file_path_tgt, 'r') as f:
#             for tgt_sent in f:
#                 if iNumLine < useful_lines[pointerSrc]:  # This line was in the src
#                     iNumLine += 1
#                     continue
#
#                 if tgt_sent not in dict_tgt:
#                     dict_tgt[tgt_sent] = 1
#                     final_useful_lines.append(iNumLine)
#                 iNumLine += 1
#                 pointerSrc += 1
#                 if pointerSrc >= len(useful_lines):
#                     break
#
#         del useful_lines
#         del dict_tgt  # Save memory
#
#         # ToDo: Sentences should be selected not randomly but based also on the similarity score. Lines are sorted
#         # following this criteria. In our case we should select only good ones.
#         for size in aSizesTrain:
#             if size > len(final_useful_lines):
#                 print(
#                 "ERROR: The size is higher than the size (%d) of the data (%d) for file (%s) and langs (%s-%s)" % (
#                 size, len(final_useful_lines), file, src, tgt))
#                 return "NO"
#
#             step = int(floor(len(final_useful_lines) / (NFOLDS * size)))
#             if step >= 1:  # there are enough sentences for NFOLDS, if not then we need to do sampling with replacement
#                 # rand_numbers = final_useful_lines[:NFOLDS*size+1]  # We target only the first lines since they are the best
#                 rand_numbers = final_useful_lines[:]  # Random selection is needed for JA-CN pairs
#                 random.shuffle(rand_numbers)
#             else:
#                 rand_numbers = []
#                 for nF in range(NFOLDS):
#                     rand_numbers = rand_numbers + random.sample(final_useful_lines, size)
#
#             for nF in range(NFOLDS):
#                 rand_numbers_n = sorted(rand_numbers[nF * size:(nF + 1) * size])
#                 train_file_out_path_src = train_file_path_src + '.' + str(size) + '.' + str(nF)
#                 train_file_out_path_tgt = train_file_path_tgt + '.' + str(size) + '.' + str(nF)
#                 t1 = datetime.datetime.now()
#                 if (not os.path.isfile(train_file_out_path_src)) or make_new is True:
#                     print("Creating " + train_file_out_path_src)
#
#                     # Create the new training files
#                     # h_file_src = codecs.open(train_file_out_path_src, 'w', 'utf-8')
#                     h_file_src = open(train_file_out_path_src, 'w')
#                     added = 0
#                     numLine = 0
#                     # with codecs.open(train_file_path_src, 'r', 'utf-8') as f:
#                     with open(train_file_path_src, 'r') as f:
#                         for i in rand_numbers_n:
#                             while (numLine <= i):
#                                 tgt_sent = f.readline()
#                                 numLine += 1
#
#                             h_file_src.writelines(tgt_sent)
#                             added = added + 1
#                             if added > size:
#                                 break
#
#                     h_file_src.close()
#
#                     if added != size:
#                         print("ERROR: added only " + str(added) + "instead of " + str(
#                             size) + " for src: " + train_file_out_path_src)
#                         return "NO"
#
#                 if (not os.path.isfile(train_file_out_path_tgt)) or make_new == True:
#                     print("Creating " + train_file_out_path_tgt)
#
#                     # Create the new training files
#                     # h_file_tgt = codecs.open(train_file_out_path_tgt, 'w', 'utf-8')
#                     h_file_tgt = open(train_file_out_path_tgt, 'w')
#
#                     added = 0
#                     numLine = 0
#                     #with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
#                     with open(train_file_path_tgt, 'r') as f:
#                         for i in rand_numbers_n:
#                             while (numLine <= i):
#                                 tgt_sent = f.readline()
#                                 numLine += 1
#
#                             h_file_tgt.writelines(tgt_sent)
#                             added = added + 1
#                             if added > size:
#                                 break
#                     h_file_tgt.close()
#                     if added != size:
#                         print("ERROR: added only " + str(added) + "instead of " + str(
#                             size) + " for tgt: " + train_file_out_path_tgt)
#                         return "NO"
#
#                 t2 = datetime.datetime.now()
#                 print("Execution time: %s" % (t2 - t1))
#
#         return "OK"
#     else:
#         print("ERROR: in " + file + " for files src: " + src + ' or tgt: ' + tgt + ' check that they exist')
#         return "NO"
#
#
# def createTrainingFiles(filesPerLanguage, overwrite_all=False):
#     print("***** Creating training files ******")
#     pool = multiprocessing.Pool(processes=NUM_CORES_MAX)
#
#     tmpTask = []
#     for (lang_pair, files) in filesPerLanguage.items():
#         print('Processing language pair ' + lang_pair)
#         aPrefix = lang_pair.split("-")
#         src = aPrefix[0]
#         tgt = aPrefix[1]
#         for file in files:
#             tmpTask.append((TRAIN_DATA_DIR + file, src, tgt, overwrite_all))
#    # try:
#         results = [pool.map(createtrainfile_wrapper, tmpTask)]
#    # except ValueError:
#       #  print("UTF8 error ")
#     if len(results[0]) != len(tmpTask):
#         print("ERROR: Creating training files")
#         exit(-1)
#     print("... Done")
#

class PreProcessingClass():
    def __init__(self, submissionsPerLanguagePerYear, filesPerLanguage, filesPerLanguageForLM, train_data_dir,
                 scripts_dir, submissions_dir, cfg):
        global SCRIPTS_DIR
        global TRAIN_DATA_DIR

        self.submissionsPerLanguagePerYear = submissionsPerLanguagePerYear
        self.filesPerLanguage = filesPerLanguage
        self.filesPerLanguageForLM = filesPerLanguageForLM
        self.train_data_dir = train_data_dir
        self.scripts_dir = scripts_dir
        self.submissions_dir = submissions_dir
        self.cfg = cfg

        SCRIPTS_DIR = scripts_dir
        TRAIN_DATA_DIR = train_data_dir
        # configuration = self.cfg
        # print(self.cfg['runall']['NFOLDS'])
        # print(configurar1['runall']['NFOLDS'])
        # print(configurar1['runall']['NFOLDS'])

    def tokenize_wrapper(self, args):
        return self.tokenizeFile(*args)

    def tokenizeFile(self, filename_in, lang, filename_out):
        cmd = 'perl ' + SCRIPTS_DIR + 'perl/tokenizer.perl -time -no-escape -threads 3 -l ' + lang + ' < ' + filename_in \
              + ' > ' + filename_out
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
        print('----' + filename_in + ' : ' + p.decode())
        # for line in p.stdout.readlines():
        # for line in p.readlines():
        #    print('----' + filename_in + ' : ' + line)
        # retval = p.wait()
        if not os.path.exists(filename_out) or os.path.getsize(filename_out) == 0:
            print('ERROR: tokenizing file ' + filename_in)
            return "NO"
        print("... Done " + filename_in + " saving in " + filename_out)
        return "OK"

    def clean_wrapper(self,args):
        return self.cleanFile(*args)

    def cleanFile(self, file, src, tgt):
        filename_out_src = file + '.token.' + src
        filename_out_tgt = file + '.token.' + tgt

        MIN_SENTENCE_LENGTH = self.cfg['preProcessModule']['MIN_SENTENCE_LENGTH']
        MAX_SENTENCE_LENGTH = self.cfg['preProcessModule']['MAX_SENTENCE_LENGTH']

        MIN_SENTENCE_LENGTH_CHARS = self.cfg['preProcessModule']['MIN_SENTENCE_LENGTH_CHARS']
        MAX_SENTENCE_LENGTH_CHARS = self.cfg['preProcessModule']['MAX_SENTENCE_LENGTH_CHARS']

        if src in lang_char_tokenization or tgt in lang_char_tokenization:
            cmd = 'perl ' + SCRIPTS_DIR + 'perl/clean-corpus-n.perl -lc ' + file + '.token ' + src + ' ' + tgt + ' ' + file \
                  + '.clean ' + str(MIN_SENTENCE_LENGTH_CHARS) + ' ' + str(MAX_SENTENCE_LENGTH_CHARS)
        else:
            cmd = 'perl ' + SCRIPTS_DIR + 'perl/clean-corpus-n.perl -lc ' + file + '.token ' + src + ' ' + tgt + ' ' + file \
                  + '.clean ' + str(MIN_SENTENCE_LENGTH) + ' ' + str(MAX_SENTENCE_LENGTH)

        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # for line in p.stdout.readlines():
        #     print('----' + file + ' > ' + line)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()[0]
        print('----' + file + ' > ' + p.decode())
        # retval = p.wait()
        if not os.path.exists(filename_out_src) or os.path.getsize(filename_out_src) == 0:
            print('ERROR: tokenizing file ' + filename_out_src)
            return "NO"
        elif not os.path.exists(filename_out_tgt) or os.path.getsize(filename_out_tgt) == 0:
            print('ERROR: tokenizing file ' + filename_out_tgt)
            return "NO"

        print("... Done " + file + " saving in " + filename_out_src + ' and ' + filename_out_tgt)
        return "OK"

    def preproc_wrapper(self, args):
        return self.preprocFile(*args)

    def preprocFile(self, filename_in, lang, filename_out):
        # cmd = 'perl ' + SCRIPTS_DIR + 'perl/replace-unicode-punctuation.perl ' + ' < ' + filename_in
        # cmd += ' | perl ' + SCRIPTS_DIR + 'perl/remove-non-printing-char.perl - '
        # cmd += ' | perl ' + SCRIPTS_DIR + 'perl/normalize-punctuation.perl -l ' + lang + ' - '
        # cmd += ' | perl ' + SCRIPTS_DIR + 'perl/lowercase.perl ' + ' - > ' + filename_out
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # for line in p.stdout.readlines():
        #     print('---- ' + filename_in + ' : ' + line)
        # retval = p.wait()
        # if not os.path.exists(filename_out) or os.path.getsize(filename_out) == 0:
        #     print('ERROR: tokenizing file ' + filename_in)
        #     return "NO"
        # print("... Done " + filename_in + " saving in " + filename_out)
        try:
            with open(filename_in, 'r') as f_in, open(filename_out, 'w') as f_out:
                for line in f_in.readlines():
                    f_out.write(line.lower())
            return "OK"
        except:
            print("There was an error when processing file %s" % filename_in)
            return "NO"

    def createtrainfile_wrapper(self, args):
        return self.createTrainingFilesParallel(*args)

    def createTrainingFilesParallel(self, file, src, tgt, make_new=False):
        print("***** Creating training file for " + file + ' src: ' + src + ' and tgt: ' + tgt + " ******")

        train_file_path_src = file + '.lower.' + src
        train_file_path_tgt = file + '.lower.' + tgt

        #access to variables that are needed in this method and charging them as local variables
        NFOLDS = self.cfg['preProcessModule']['NFOLDS']
        aSizesTrain = [self.cfg['preProcessModule']['aSizesTrain']]  # For WOT2016-JA-HI los corchetes son necesarios por alguna razon

        if os.path.isfile(train_file_path_src) and os.path.isfile(train_file_path_tgt):
            # if (make_new is True) :
            # First step is to read the src and load unique lines
            dict_src = {}
            useful_lines = []
            iNumLine = 0
            # with codecs.open(train_file_path_src, 'r', 'utf-8', errors='ignore') as f: buffering=None, encoding=None, errors=None, newline=None, closefd=True
            with open(train_file_path_src, 'r', encoding='utf-8', errors='ignore') as f:
                for src_sent in f:
                    # src_sent= unicode(src_sent,errors='ignore')
                    if src_sent not in dict_src:
                        dict_src[src_sent] = 1
                        useful_lines.append(iNumLine)  # Save the line number
                    iNumLine += 1
            # len_src = len(useful_lines)
            del dict_src  # Save memory

            dict_tgt = {}
            iNumLine = 0
            pointerSrc = 0
            final_useful_lines = []
            # with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
            with open(train_file_path_tgt, 'r') as f:
                for tgt_sent in f:
                    if iNumLine < useful_lines[pointerSrc]:  # This line was in the src
                        iNumLine += 1
                        continue

                    if tgt_sent not in dict_tgt:
                        dict_tgt[tgt_sent] = 1
                        final_useful_lines.append(iNumLine)
                    iNumLine += 1
                    pointerSrc += 1
                    if pointerSrc >= len(useful_lines):
                        break

            del useful_lines
            del dict_tgt  # Save memory

            # ToDo: Sentences should be selected not randomly but based also on the similarity score. Lines are sorted
            # following this criteria. In our case we should select only good ones.
            for size in aSizesTrain:
                if size > len(final_useful_lines):
                    print(
                        "ERROR: The size is higher than the size (%d) of the data (%d) for file (%s) and langs (%s-%s)" % (
                            size, len(final_useful_lines), file, src, tgt))
                    return "NO"

                step = int(floor(len(final_useful_lines) / (NFOLDS * size)))
                if step >= 1:  # there are enough sentences for NFOLDS, if not then we need to do sampling with replacement
                    # rand_numbers = final_useful_lines[:NFOLDS*size+1]  # We target only the first lines since they are the best
                    rand_numbers = final_useful_lines[:]  # Random selection is needed for JA-CN pairs
                    random.shuffle(rand_numbers)
                else:
                    rand_numbers = []
                    for nF in range(NFOLDS):
                        rand_numbers = rand_numbers + random.sample(final_useful_lines, size)

                for nF in range(NFOLDS):
                    rand_numbers_n = sorted(rand_numbers[nF * size:(nF + 1) * size])
                    train_file_out_path_src = train_file_path_src + '.' + str(size) + '.' + str(nF)
                    train_file_out_path_tgt = train_file_path_tgt + '.' + str(size) + '.' + str(nF)
                    t1 = datetime.datetime.now()
                    if (not os.path.isfile(train_file_out_path_src)) or make_new is True:
                        print("Creating " + train_file_out_path_src)

                        # Create the new training files
                        # h_file_src = codecs.open(train_file_out_path_src, 'w', 'utf-8')
                        h_file_src = open(train_file_out_path_src, 'w')
                        added = 0
                        numLine = 0
                        # with codecs.open(train_file_path_src, 'r', 'utf-8') as f:
                        with open(train_file_path_src, 'r') as f:
                            for i in rand_numbers_n:
                                while (numLine <= i):
                                    tgt_sent = f.readline()
                                    numLine += 1

                                h_file_src.writelines(tgt_sent)
                                added = added + 1
                                if added > size:
                                    break

                        h_file_src.close()

                        if added != size:
                            print("ERROR: added only " + str(added) + "instead of " + str(
                                size) + " for src: " + train_file_out_path_src)
                            return "NO"

                    if (not os.path.isfile(train_file_out_path_tgt)) or make_new == True:
                        print("Creating " + train_file_out_path_tgt)

                        # Create the new training files
                        # h_file_tgt = codecs.open(train_file_out_path_tgt, 'w', 'utf-8')
                        h_file_tgt = open(train_file_out_path_tgt, 'w')

                        added = 0
                        numLine = 0
                        # with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
                        with open(train_file_path_tgt, 'r') as f:
                            for i in rand_numbers_n:
                                while (numLine <= i):
                                    tgt_sent = f.readline()
                                    numLine += 1

                                h_file_tgt.writelines(tgt_sent)
                                added = added + 1
                                if added > size:
                                    break
                        h_file_tgt.close()
                        if added != size:
                            print("ERROR: added only " + str(added) + "instead of " + str(
                                size) + " for tgt: " + train_file_out_path_tgt)
                            return "NO"

                    t2 = datetime.datetime.now()
                    print("Execution time: %s" % (t2 - t1))

            return "OK"
        else:
            print("ERROR: in " + file + " for files src: " + src + ' or tgt: ' + tgt + ' check that they exist')
            return "NO"

    def createTrainingFiles(self, filesPerLanguage, overwrite_all=False):
        print("***** Creating training files ******")
        pool = multiprocessing.Pool(processes= self.cfg['preProcessModule']['NUM_CORES_MAX'])

        tmpTask = []
        for (lang_pair, files) in filesPerLanguage.items():
            print('Processing language pair ' + lang_pair)
            aPrefix = lang_pair.split("-")
            src = aPrefix[0]
            tgt = aPrefix[1]
            for file in files:
                tmpTask.append((TRAIN_DATA_DIR + file, src, tgt, overwrite_all))
            # try:
            results = [pool.map(self.createtrainfile_wrapper, tmpTask)]
        # except ValueError:
        #  print("UTF8 error ")
        if len(results[0]) != len(tmpTask):
            print("ERROR: Creating training files")
            exit(-1)
        print("... Done")





    def tokenize(self, list_files_tokenize):
        print("***** Tokenizing files ******")
        t1 = datetime.datetime.now()
        print("Starting at " + str(t1))
        # pool = multiprocessing.Pool(processes=NUM_CORES_MAX)
        # print(self.cfg['preProcessModule']['NUM_CORES_MAX'])
        pool = multiprocessing.Pool(processes= self.cfg['preProcessModule']['NUM_CORES_MAX'])
        results = [pool.map(self.tokenize_wrapper, list_files_tokenize)]
        for res in results[0]:
            if res == "NO":
                print("ERROR: One of the tokenization files could not be created.")
                exit(-1)
        t2 = datetime.datetime.now()
        print("... Done tokenization in: %s " % (t2-t1))

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
        print("... Done cleaning in: %s " % (t2-t1))

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


    def preprocess_files(self, bDoAll=False):
        # Perform pre-processing for all files in a give directory
        files_to_tokenize = []
        files_to_clean = []
        files_to_preprocess = []

        # First the parallel corpus
        print('Detecting files to pre-process')
        for (lang_pair, files) in self.filesPerLanguage.items():
            print('Processing language pair ' + lang_pair)
            aPrefix = lang_pair.split("-")
            src = aPrefix[0]
            tgt = aPrefix[1]

            for file in files:
                file_src = self.train_data_dir + file + '.' + src
                if os.path.exists(file_src):
                    filename_token = self.train_data_dir + file + '.token.' + src
                    if not os.path.exists(filename_token) or bDoAll is True:
                        print('Adding file ' + file_src + ' for tokenization')
                        files_to_tokenize.append((file_src, src, filename_token))

                    filename_clean = self.train_data_dir + file + '.clean.' + src
                    filename_proc = self.train_data_dir + file + '.lower.' + src
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_src + ' for preprocessing')
                        files_to_preprocess.append((filename_clean, src, filename_proc))
                else:
                    print('ERROR: file: ' + file_src + ' does not exists.')

                file_tgt = self.train_data_dir + file + '.' + tgt
                if os.path.exists(file_tgt):
                    filename_token = self.train_data_dir + file + '.token.' + tgt
                    if not os.path.exists(filename_token) or bDoAll is True:
                        print('Adding file ' + file_tgt + ' for tokenization')
                        files_to_tokenize.append((file_tgt, tgt, filename_token))

                    filename_proc = self.train_data_dir + file + '.lower.' + tgt
                    filename_clean = self.train_data_dir + file + '.clean.' + tgt
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_tgt + ' for preprocessing')
                        files_to_preprocess.append((filename_clean, tgt, filename_proc))
                else:
                    print('ERROR: file: ' + file_tgt + ' does not exists.')

                if not os.path.exists(self.train_data_dir + file + '.clean.' + src) or \
                        not os.path.exists(self.train_data_dir + file + '.clean.' + tgt) \
                        or bDoAll is True:
                    print('Adding files ' + self.train_data_dir + file + ' for cleaning')
                    files_to_clean.append((self.train_data_dir + file, src, tgt))

        # Files for LMs
        for (lang, files) in self.filesPerLanguageForLM.items():
            print('Processing language ' + lang)
            for file in files:
                file_for_lm = self.train_data_dir + file + '.' + lang
                if os.path.exists(file_for_lm):
                    filename_token = self.train_data_dir + file + '.token.' + lang
                    if not os.path.exists(filename_token) or bDoAll is True:
                        print('Adding file ' + file_for_lm + ' for tokenization')
                        files_to_tokenize.append((file_for_lm, lang, filename_token))

                    filename_proc = self.train_data_dir + file + '.lower.' + lang
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_for_lm + ' for preprocessing')
                        files_to_preprocess.append((filename_token, lang, filename_proc))

                else:
                    print('ERROR-LM: file: ' + file_for_lm + ' does not exists.')

        # Files for Submissions
        for (lang_pair, years) in self.submissionsPerLanguagePerYear.items():
            aPrefix = lang_pair.split("-")
            tgt = aPrefix[1]
            for (year, info) in self.submissionsPerLanguagePerYear[lang_pair].items():
                print('Processing language ' + lang_pair + ' for year: ' + year)

                # Add the reference file
                reference_file = info['reference']
                filename_token = self.submissions_dir + reference_file + '.token'
                if not os.path.exists(filename_token) or bDoAll is True:
                    print('Adding file ' + reference_file + ' for tokenization')
                    files_to_tokenize.append((self.submissions_dir + reference_file, tgt, filename_token))

                filename_proc = self.submissions_dir + reference_file + '.lower'
                if not os.path.exists(filename_proc) or bDoAll is True:
                    print('Adding file ' + reference_file + ' for preprocessing')
                    files_to_preprocess.append((filename_token, tgt, filename_proc))

                source_file = info['source']
                filename_token = self.submissions_dir + source_file + '.token'
                if not os.path.exists(filename_token) or bDoAll is True:
                    print('Adding file ' + source_file + ' for tokenization')
                    files_to_tokenize.append((self.submissions_dir + source_file, src, filename_token))

                filename_proc = self.submissions_dir + source_file + '.lower'
                if not os.path.exists(filename_proc) or bDoAll is True:
                    print('Adding file ' + source_file + ' for preprocessing')
                    files_to_preprocess.append((filename_token, src, filename_proc))

                for filename in info['submissions']:
                    submission_file = self.submissions_dir + filename
                    if os.path.exists(submission_file):
                        filename_token = self.submissions_dir + filename + '.token'
                        if not os.path.exists(filename_token) or bDoAll is True:
                            print('Adding file ' + submission_file + ' for tokenization')
                            files_to_tokenize.append((self.submissions_dir + filename, tgt, filename_token))

                        filename_proc = self.submissions_dir + filename + '.lower'
                        if not os.path.exists(filename_proc) or bDoAll is True:
                            print('Adding file ' + submission_file + ' for preprocessing')
                            files_to_preprocess.append((filename_token, tgt, filename_proc))
                    else:
                        print('ERROR: file: ' + submission_file + ' does not exists.')

        # Perform tokenization
        if len(files_to_tokenize) > 0:
            self.tokenize(files_to_tokenize)
        else:
            print('Nothing to tokenize')

        # Cleaning files
        if len(files_to_clean) > 0:
            self.clean(files_to_clean)
        else:
            print('Nothing to clean')

        # Perform preprocessing
        if len(files_to_preprocess) > 0:
            self.preProcessing(files_to_preprocess)
        else:
            print('Nothing to preprocess')

        print('Finished')


def main(): ###Aqui deberimos buscar al forma de meter las variables que se usan en general en los metodos, ademas estas variables cuando lsa funciones son invcoadas desde el runall deben de pasar los parametros??

    parser = argparse.ArgumentParser()
    parser.add_argument('-overwrite', help='Overwrite all the files.', action='store_true')

    parser.add_argument('-c', '--my-config', type=str, dest='MyConfigFilePath', required=False, is_config_file=False,
                        help='config file path')
    parser.add_argument('-d', '--num_cores', dest='NofCores', help='Number of cores',
                        env_var='NUM_CORES')  # this option can be set in a config file because it starts with '--'

    args = parser.parse_args()

    filepath = args.MyConfigFilePath

    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # args = parser.parse_args()
    overwrite_all = args.overwrite
    # submissionsPerLanguagePerYear, filesPerLanguage, filesPerLanguageForLM, train_data_dir,
    #                  scripts_dir, submissions_dir, cfg
    clPrep = PreProcessingClass()

    clPrep.preprocess_files(bDoAll=overwrite_all)
    clPrep.createTrainingFiles(filesPerLanguage, overwrite_all=overwrite_all)

if __name__ == '__main__':
    main()