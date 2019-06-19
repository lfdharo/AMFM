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




class PreProcessingClass():
    """ Class that holds all the necessary parameters for preProcessing the files for Train purposes"""

    def __init__(self, submissionsPerLanguagePerYear, filesPerLanguage, filesPerLanguageForLM, train_data_dir,
                 scripts_dir, submissions_dir, cfg):
        """

        :param submissionsPerLanguagePerYear: This is currently not being used for this implementation
        :param filesPerLanguage: Files seted up in the configuration file "yaml" for training the SVD
        :param filesPerLanguageForLM: Files setted up in the configuration file "yaml" for training the LM
        :param train_data_dir: Directory for the train data it is related with the cfg INPUT
        :param scripts_dir: directory where the scripts if needed are
        :param submissions_dir: directory for the Train process
        :param cfg: path for your configuration file
        """
        global SCRIPTS_DIR
        global TRAIN_DATA_DIR

        self.submissionsPerLanguagePerYear = submissionsPerLanguagePerYear
        self.filesPerLanguage = filesPerLanguage
        self.filesPerLanguageForLM = filesPerLanguageForLM
        self.train_data_dir = train_data_dir
        self.scripts_dir = scripts_dir
        self.submissions_dir = submissions_dir
        self.cfg = cfg
        self.lang_char_tokenization = cfg['preProcessModule']['lang_char_tokenization']
        self.file_out = cfg['directories']['OUTPUT']

        self.sc = set(['-', "'","%"])
        self.to_remove = ''.join([c for c in string.punctuation if c not in self.sc])
        self.to_remove= self.to_remove +"“"+"”" #We had some extra characters to remove from texts
        self.table = dict((ord(char), u' ') for char in self.to_remove) #We replace the characters in the list(strg) with a blanc space = u' ' ( unicode blanc space)

        self.sc = set([',', '!', '?', '.'])
        self.to_separate = ''.join([c for c in string.punctuation if c not in self.sc])
        self.table_separate = dict((ord(char), u' ' + char + u' ') for char in self.to_separate) # We separate the characters in the list(strg) with a blanc space = u' ' ( unicode blanc space)

        SCRIPTS_DIR = scripts_dir
        TRAIN_DATA_DIR = train_data_dir

    def preProcessWork(self, sentence, lang):
        """"
        Basic method to perform basic Tokenization line by line
        """

        if len(sentence) == 0:  # To avoid empty lines
            return '_EMPTY_'

        # Remove some punctuation
        s = sentence.translate(self.table)
        if lang == 'my':  # we need to remove separation for | and -
            for k, v in self.table_separate.iteritems():
                if v == u' |' or v == u' -':
                    self.table_separate[k] = v.strip()
        # Replaces especial characters in to " '@%' " = " ' @ % ' "
        s = s.translate(self.table_separate)

        # Tokenization by characters for most of Asian languages except for English, Indian, Korean and Myanmar

        if lang != 'en' and lang != 'in' and lang != 'ko'  and lang != 'my': #and lang != 'ta':and lang != 'hi'
            tokens = [' '.join([c for c in list(word.strip())]) for word in s.split()]

        else: #other langauges such as Chinese, Japanese, Hindi and Tamil
            tokens = s.split()
            if lang == 'my':  # We need to replace the | for -
                tokens = [x.replace('|', '-') for x in tokens]

        s = ' '.join(tokens)
        return s


    def tokenize(self, list_files_tokenize):
        """
        Method that initalizes the tokenization process
        :param list_files_tokenize: list of files that we want to get tokenized
        :return:
        """
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
        """
        Wrapper for the tokenizer
        """
        return self.tokenizeFile(*args)

    def tokenizeFile(self, filename_in, lang, filename_out):
        """
        Method that gets as input the files that has to be tokenized and passes each sentence of thsi file to the tokenizer
        :param filename_in: File path that has to be tokenized
        :param lang: Language in which the file is
        :param filename_out: File path for the outputs
        :return:
        """
        if self.cfg['preProcessModule']['Tokenize'] == True: #si queremos ejecutar la tokenizacion
            with open(filename_in, 'r') as f_in, open(filename_out, 'w',encoding='utf-8') as f_out:
                for line in f_in: #f_in.readlines():
                     linetokenized = self.preProcessWork(line,lang)
                     f_out.write(linetokenized+'\n')

        else:
            cmd = self.cfg['preProcessModule']['Tokenize_Script']  #insert your own line for tokenizing files

        if not os.path.exists(filename_out) or os.path.getsize(filename_out) == 0:
            print('ERROR: tokenizing file ' + filename_in)
            return "NO"
        print("... Done " + filename_in + " saving in " + filename_out)
        return "OK"

    ##################################################


    ######### THIRD STEP OF PREPROCESSING ############
    def clean(self, list_files_clean):
        """
        Method of the preporcess incharge of reciving as inputs the files already tokenized and lower cased and applies some restrictions to the sentences forming the file depending on the language
        :param list_files_clean: List of files that we want to clean
        :return:
        """
        print("***** cleaning files ******")
        t1 = datetime.datetime.now()
        print("Starting at " + str(t1))
        pool = multiprocessing.Pool(processes=self.cfg['preProcessModule']['NUM_CORES_MAX'])
        results = [pool.map(self.clean_wrapper, list_files_clean)]
        for res in results[0]:
            if res == "NO":
                print("ERROR: One of the cleaning files could not be created.")
                exit(-1)
        t2 = datetime.datetime.now()
        print("... Done cleaning in: %s " % (t2 - t1))

    def clean_wrapper(self, args):
        """
                Wrapper for the cleaner
                """
        return self.cleanFile(*args)

    def cleanFile(self, file, src, tgt):
        """
        Method in charged of cleaning the files,as input recieves the output from the last preprocessing step in this case lowe cassing nad applies restrictions to sentence length depending on their language, this done to not get empty
        sentences or not getting sentences that are way to long
        :param file: path of the file that has to be clean
        :param src: language of one of the files
        :param tgt: language of one of the files
        :return:
        """
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
        """
        Method used for another step of preprocessing in this case lower casing the files
        :param list_files_lowercase: list of files that need to get lower cased
        :return:
        """
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
        """
                      Wrapper for the lower case
                      """
        return self.preprocFile(*args)

    def preprocFile(self, filename_in, lang, filename_out):
        """
        Method in charged explicitly of the lower casing, gets as input the file that has to be lower and also the path for writting the results of this process
        :param filename_in: path of the files
        :param lang: language in which the file is
        :param filename_out:path of the output file
        :return:
        """
        try:
            with open(filename_in, 'r') as f_in, open(filename_out, 'w',encoding='utf-8') as f_out:
                for line in f_in.readlines():
                    f_out.write(line.lower())
            return "OK"
        except:
            print("There was an error when processing file %s" % filename_in)
            return "NO"

    ##################################################


    ########## PREPROCESS FILES #################

    def preprocess_files(self, bDoAll=False):
        """
        Prepares and sets up directories for the preprocessing process
        :param bDoAll:
        :return:
        """
        # Perform pre-processing for all files in a give directory
        files_to_tokenize = []
        files_to_clean = []
        files_to_preprocess = []

        # Creates the specfic File in "Files_input" for storing all the out put files related to each Language
        if not os.path.exists(self.file_out):
            print("...creating " + self.file_out)
            os.makedirs(self.file_out)

        # Creates the specfic File in "Files_output" for storing all the out put files related to each Language
        if not os.path.exists(self.train_data_dir):
            print("...creating " + self.train_data_dir)
            os.makedirs(self.train_data_dir)


        # First the parallel corpus
        print('Detecting files to pre-process')
        for (lang_pair, files) in self.filesPerLanguage.items(): #en este caso se ejecuta primero para la primera entrada que tenemos que sera jp-en JIJI corpus y despues en-jp...
            print('Processing language pair ' + lang_pair)
            aPrefix = lang_pair.split("-")
            src = aPrefix[0]
            tgt = aPrefix[1]

            corpus = files[0].split('/')

            # Creates the specfic File in "Files_output" for storing all the out put files related to each Language
            if not os.path.exists(self.file_out + str(corpus[0])):
                print("...creating " + self.file_out + str(corpus[0]))
                os.makedirs(self.file_out + str(corpus[0]))

            for file in files:
                file_src = self.train_data_dir + file + '.' + src
                if os.path.exists(file_src):
                    filename_token = self.file_out + file + '.token.' + src
                    if not os.path.exists(filename_token) or bDoAll is True:
                        print('Adding file ' + file_src + ' for tokenization list')
                        files_to_tokenize.append((file_src, src, filename_token)) #We add to the list the files that we want to apply tokenization

                    filename_proc = self.file_out + file + '.lower.' + src
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_src + ' for preprocessing list')
                        files_to_preprocess.append((filename_token, src, filename_proc)) #añadimos a la lista de preprocess= lowerCasing los archivos de clean(deberian ser los archivos de token

                    filename_clean = self.file_out + file + '.clean.' + src
                    if not os.path.exists(filename_clean) or bDoAll is True:
                        print('Adding file ' + file_src + ' for cleaning list')
                        files_to_clean.append((self.file_out + file, src, tgt))

                else:
                    print('ERROR: file: ' + file_src + ' does not exists.')

                file_tgt = self.train_data_dir + file + '.' + tgt
                if os.path.exists(file_tgt):
                    filename_token = self.file_out + file + '.token.' + tgt
                    if not os.path.exists(filename_token) or bDoAll is True:
                        print('Adding file ' + file_tgt + ' for tokenization list')
                        files_to_tokenize.append((file_tgt, tgt, filename_token))

                    filename_proc = self.file_out + file + '.lower.' + tgt
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_tgt + ' for preprocessing list')
                        files_to_preprocess.append((filename_token, tgt, filename_proc))

                    filename_clean = self.file_out + file + '.clean.' + tgt
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_tgt + ' for cleaning list')
                        files_to_clean.append((self.file_out + file, src, tgt))
                else:
                    print('ERROR: file: ' + file_tgt + ' does not exists.')

        # Files for LMs
        for (lang, files) in self.filesPerLanguageForLM.items():
            print('Processing language ' + lang)
            for file in files:
                file_for_lm = self.train_data_dir + file + '.' + lang
                if os.path.exists(file_for_lm):
                    filename_token = self.file_out + file + '.token.' + lang
                    if not os.path.exists(filename_token) or bDoAll is True:
                        print('Adding file ' + file_for_lm + ' for tokenization list')
                        files_to_tokenize.append((file_for_lm, lang, filename_token))

                    filename_proc = self.file_out + file + '.lower.' + lang
                    if not os.path.exists(filename_proc) or bDoAll is True:
                        print('Adding file ' + file_for_lm + ' for preprocessing list')
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
##################################################################################################################################

        # Perform tokenization      1
        if len(files_to_tokenize) > 0:
            self.tokenize(files_to_tokenize)

        else:
            print('Nothing to tokenize')

         # Perform preprocessing     2 el orden lo he cambiado 14/03/2019 el 2 por el 3 que es el orden mas logico cuidado al ejecutar
        if len(files_to_preprocess) > 0:
            self.preProcessing(files_to_preprocess)
        else:
            print('Nothing to preprocess')

        # Cleaning files            3
        if len(files_to_clean) > 0:
            self.clean(files_to_clean)
        else:
            print('Nothing to clean')



        print('Finished')

    ##################################################

    ########## CREATE TRAINING FILES #################

    def createtrainfile_wrapper(self, args):
        return self.createTrainingFilesParallel(*args)

    def createTrainingFilesParallel(self, file, src, tgt, make_new=False):
        """
        Methos in charge of creating the files according to the configuration see yaml file
        :param file: file path that we want to use for creating the file for training
        :param src: src language
        :param tgt: tgt language
        :param make_new:
        :return:
        """
        print("***** Creating training file for " + file + ' src: ' + src + ' and tgt: ' + tgt + " ******")

        train_file_path_src = file + '.lower.' + src
        train_file_path_tgt = file + '.lower.' + tgt

        #access to variables that are needed in this method and charging them as local variables
        NFOLDS = self.cfg['preProcessModule']['NFOLDS']
        aSizesTrain = [self.cfg['preProcessModule']['aSizesTrain']]

        if os.path.isfile(train_file_path_src) and os.path.isfile(train_file_path_tgt):
            # if (make_new is True) :
            # First step is to read the src and load unique lines
            dict_src = {}
            useful_lines = []
            iNumLine = 0
            # with codecs.open(train_file_path_src, 'r', 'utf-8', errors='ignore') as f: buffering=None, encoding=None, errors=None, newline=None, closefd=True
            with open(train_file_path_src, 'r', errors='ignore') as f:
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
            with open(train_file_path_tgt, 'r', errors='ignore') as f:
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
                        h_file_src = open(train_file_out_path_src, 'w',encoding='utf-8')
                        added = 0
                        numLine = 0
                        # with codecs.open(train_file_path_src, 'r', 'utf-8') as f:
                        with open(train_file_path_src, 'r', errors='ignore') as f:
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
                        h_file_tgt = open(train_file_out_path_tgt, 'w',encoding='utf-8')

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
        """
        Method in charge of creating the files in the way that the training process needs it, this means the way to write the file name as well as what the file must contain
        :param filesPerLanguage: File path for the files that we want to use for training
        :param overwrite_all:
        :return:
        """
        print("***** Creating training files ******")
        pool = multiprocessing.Pool(processes= self.cfg['preProcessModule']['NUM_CORES_MAX'])

        tmpTask = []
        for (lang_pair, files) in filesPerLanguage.items():
            print('Processing language pair ' + lang_pair)
            aPrefix = lang_pair.split("-")
            src = aPrefix[0]
            tgt = aPrefix[1]
            for file in files:
                tmpTask.append((self.file_out + file, src, tgt, overwrite_all))
            results = [pool.map(self.createtrainfile_wrapper, tmpTask)]

        if len(results[0]) != len(tmpTask):
            print("ERROR: Creating training files")
            exit(-1)
        print("... Done")

    ##################################################

def main():

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

    root_dir = cfg['directories']['INPUT']

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
