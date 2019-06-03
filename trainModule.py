__author__ = 'luisdhe' , 'vanmaren'
import os
import subprocess
import multiprocessing
from multiprocessing import Pool
# We must import this explicitly, it is not imported by the top-level
# multiprocessing module.
import multiprocessing.pool
from vector_space import VectorSpace
import codecs
import argparse
import datetime
import random
import yaml
from math import floor

# with open('/home/enrique/Escritorio/TFG_Pendrive/AMFM/Settings.yaml', 'r') as ymlfile:
#     cfg = yaml.load(ymlfile)

#type_vectorizer = 'tfidf'
type_vectorizer = 'counts'

root_dir = os.path.dirname(os.path.realpath(__file__))
train_data_dir = root_dir + '/'
dir_lm_out = root_dir + '/lms/' #Aqui elejimos donde se va buscar y volcar las salidas de los LM mas o menos
dir_svd_mono = root_dir + '/svd_mono_' + type_vectorizer + '/'

aTypeLingualExp = ['mono']
overwrite_all = False
#
# NFOLDS = cfg['trainModule']['NFOLDS']  # Number of NFOLDS cross-training sets we are creating for the AM experiments
# MIN_COUNTS = cfg['trainModule']['MIN_COUNTS'] # Number of times a word must occur to be included in the SVM model
# MIN_NGRAM_ORDER = cfg['trainModule']['MIN_NGRAM_ORDER']
# MAX_NGRAM_ORDER = cfg['trainModule']['MAX_NGRAM_ORDER']
# NUM_MAX_CORES = cfg['trainModule']['NUM_MAX_CORES']
# # NUM_MAX_CORES = 4
# # final_size = 21080
# dictSizesTrain = {cfg['trainModule']['dictSizesTrain']['Max']: cfg['trainModule']['dictSizesTrain']['Min']}  # For WAT2016-JA-HI
# num_cores_mono = {cfg['trainModule']['num_cores_mono']['Max']: cfg['trainModule']['num_cores_mono']['Min']}


filesPerLanguageForLM = {
    'en': [
        'ASPEC/ASPEC-JE/train/train',
    ],
    'jp': [
        'ASPEC/ASPEC-JE/train/train',
    ],
}

filesPerLanguage = {
    'en-en': [
        'train-1.en',
    ],
    'jp-jp': [
        'train-1.jp',
    ],
}


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

#
# def createOutputDirs():
#     print("***** Creating output directories *****")
#
#     if not os.path.exists(dir_lm_out):
#         print("...creating " + dir_lm_out)
#         os.makedirs(dir_lm_out)
#
#     if not os.path.exists(dir_svd_mono):
#         print("...creating " + dir_svd_mono)
#         os.makedirs(dir_svd_mono)
#
#     print("... Done")
#
#
#
# def createLM_wrapper(args):
#     return createLM(*args)
#
# # Todo: Create a single text file with all the filenames or do it manually with all the training data
# def createLM(train_data_dir, lang, filename,cfg, bRetrain=False):
#     # This function extract ngram counts for a given file. If the lm already exists the system skips its creation
#     print("***** Extract n-gram counts for language " + lang + " from file: " + filename + " ******")
#
#     txt_file = train_data_dir + '/' + filename + '.' + 'lower' + '.' + lang
#     # if lang != 'en':  # We need first to tokenize in characters
#     #     txt_file = train_data_dir + '/' + filename + '.' + lang + '.lower_' + lang
#     #     with codecs.open(train_data_dir + '/' + filename + '.' + lang + '.lower', 'r', 'utf-8') as f, \
#     #             codecs.open(train_data_dir + '/' + filename + '.' + lang + '.lower_' + lang, 'w', 'utf-8') as o:
#     #         for line in f.readlines():
#     #             n_l = ' '.join(list(line.strip()))
#     #             o.write(n_l + '\n')
#
#     # MIN_NGRAM_ORDER = cfg['trainModule']['MIN_NGRAM_ORDER']
#     # MAX_NGRAM_ORDER = cfg['trainModule']['MAX_NGRAM_ORDER']
#
#
#     path_file = os.path.dirname(filename)
#     # Train language models
#     if not os.path.exists(dir_lm_out + path_file + '/' + lang):
#         print ("...creating " + dir_lm_out + path_file + '/' + lang)
#         os.makedirs(dir_lm_out + path_file + '/' + lang)
#
#     for ngram_order in range(MIN_NGRAM_ORDER, MAX_NGRAM_ORDER + 1, 1):
#         lm_file_out = dir_lm_out + path_file + '/' + lang + '/' + os.path.basename(filename) + '.' + lang + '.' + str(ngram_order) + '.lm'
#         if not os.path.exists(lm_file_out) or bRetrain is True:
#             if not os.path.exists(txt_file):
#                 print('ERROR: file ' + txt_file + ' does not exists')
#                 exit(-1)
#
#             print("Creating n-gram counts for file %s and order %d" % (filename, ngram_order))
#             cmd = 'ngram-count -unk -prune 1e-6 -minprune '+ str(ngram_order) + ' -order ' + str(ngram_order) + ' -text ' + txt_file + ' -lm ' + lm_file_out
#             print (cmd)
#             p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
#             for line in p.stdout.readlines():
#                 print('---- ' + filename + ' : ' + str(line))
#             retval = p.wait()
#
#     print("... Done " + lang + ' file: ' + filename)
#     return "DONE"
#
#
# def monosvd_wrapper(args):
#     return createMonolingualSVDforTaskParallel(*args)
#
#
# def createMonolingualSVDforTaskParallel(train_data_dir, file, size_file, size_svd, nF, lang, retrain_svd):
#     print( "***** Creating Monolingual SVD for file " + file + ' (' + lang + ") and Fold " + str(nF) + " ******")
#     t1 = datetime.datetime.now()
#     print("Starting " + file + ' (' + lang + ") and Fold " + str(nF) + ' at ' + str(t1))
#     #train_file_path = train_data_dir + file + '.lower.' + lang + '.' + str(size) + '.' + str(nF)
#     if lang != 'en' and lang != 'ko' and lang != 'hi':
#         train_file_path = train_data_dir + file + '.lower.' + lang + '.' + str(size_file) + '.' + str(nF)
#     else:
#         train_file_path = train_data_dir + file + '.lower.' + lang + '.' + str(size_file) + '.' + str(nF)
#     #train_file_path = train_data_dir + file + '.lower'
#
#     #svd_output_matrix = dir_svd_mono + '/' + file + '.' + lang + '.' + str(size) + '.' + str(nF) + '.mc' + str(MIN_COUNTS) # This is the full matrix without any reduction
#     # This is the full matrix without any reduction
#     svd_output_matrix = dir_svd_mono + '/' + file + '.' + lang + '.' + str(size_file) + '.' + str(size_svd) + '.' + str(nF)
#
#     if os.path.isfile(train_file_path):
#         if (not os.path.isfile(svd_output_matrix + '.h5') or not os.path.isfile(svd_output_matrix + '.dic')) \
#                 or overwrite_all is True or retrain_svd is True:
#             print('Reading training data from ' + train_file_path)
#             train_data = codecs.open(train_file_path, 'r', 'utf-8')
#             train_sentences = train_data.readlines()
#             train_data.close()
#
#             # Create the new training files
#             vs = VectorSpace()
#             # Trains the full matrix, later we will use the reduced dimensions
#             vs.trainVectorSpace((size_svd), lang, MIN_COUNTS, train_sentences, transforms=type_vectorizer) #Names of the output files
#             vs.save(svd_output_matrix)
#     else:
#         print("ERROR: file " + train_file_path + ' does not exist.')
#         return "NO"
#
#     t2 = datetime.datetime.now()
#     print("... Done " + file + ' (' + lang + ") and Fold " + str(nF) + ' in ' + str(t2-t1))
#     return "OK"
#
#
# def createMonolingualSVDFromTrainingFiles(train_data_dir, filesPerLanguage, size_file, size_svd, nF, retrain_svd):
#     print("***** Creating Monolingual SVD files for size " + str(size_svd) + " and Fold " + str(nF) + " ******")
#     t1 = datetime.datetime.now()
#     print("Starting at " + str(t1))
#
#     pool = multiprocessing.Pool(processes=num_cores_mono[size_svd])
#     tmpTask = []
#     for (lang_pair, files) in filesPerLanguage.items():
#         print('Processing language pair ' + lang_pair)
#         aPrefix = lang_pair.split("-")
#         tgt = aPrefix[1]
#         for file in files:
#             tmpTask.append((train_data_dir, file, size_file, size_svd, nF, tgt, retrain_svd))
#             path_file = os.path.dirname(file)
#             if not os.path.exists(dir_svd_mono + path_file):
#                 print("...creating " + dir_svd_mono + path_file)
#                 os.makedirs(dir_svd_mono + path_file)
#
#     results = [pool.map(monosvd_wrapper, tmpTask)]
#
#     for res in results[0]:
#         if res == "NO":
#             print("ERROR: One of the mono SVD matrices could not be created.")
#             exit(-1)
#
#     t2 = datetime.datetime.now()
#     print("... Done size " + str(size_svd))
#     print("Execution time: %s" % (t2-t1))
#
#     pool.close()
#     pool.join()
#     print("Finish")
#
#
# def fntTrainLMs(train_data_dir, filesPerLanguageForLM, overwrite_all=False):
# #    pool = MyPool(NUM_MAX_CORES)
#     number_of_workers = NUM_MAX_CORES
# #     number_of_workers = cfg['trainModule']['NUM_MAX_CORES']
#     with Pool(number_of_workers) as p:
#         tmpTask = []
#         for (lang, files) in filesPerLanguageForLM.items():
#             for file in files:
#                 tmpTask.append((train_data_dir, lang, file, cfg, overwrite_all))
#
#         # results = [p.map(createLM_wrapper, tmpTask)]
#         results = [p.starmap(createLM, tmpTask)]
#         for res in results[0]:
#             if res == "NO":
#                 print("ERROR: One of the LM could not be created.")
#                 exit(-1)
#
#     # pool.close()
#     # pool.join()
#     print("Finish")
#
#
# def fntCreateSVDs(train_data_dir, filesPerLanguage, dict_train_sizes, nFs, args):
#     global aTypeLingualExp
#
#     for typeLingualExp in aTypeLingualExp:
#         print("*********************************************************************")
#         print("*********************** " + typeLingualExp + " **********************")
#         print("*********************************************************************")
#
#         if typeLingualExp == "mono":
#             for (train_size, svd_size) in dict_train_sizes.items():
#                 for nF in range(nFs):
#                     createMonolingualSVDFromTrainingFiles(train_data_dir, filesPerLanguage, train_size, svd_size, nF, args.ret_svd)
#
#
# def createtrainfile_wrapper(args):
#     return createTrainingFilesParallel(*args)
#
#
# def createTrainingFilesParallel(file, src, tgt, make_new=False):
#     print("***** Creating training file for " + file + ' src: ' + src + ' and tgt: ' + tgt + " ******")
#
#     if tgt != 'en' and tgt != 'ko' and tgt != 'hi':
#         train_file_path_tgt = file + '.lower_' + tgt
#     else:
#         train_file_path_tgt = file + '.lower'
#
#     if os.path.isfile(train_file_path_tgt):
#         dict_tgt = {}
#         iNumLine = 0
#         final_useful_lines =[]
#         with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
#             for tgt_sent in f:
#                 if tgt_sent not in dict_tgt:
#                     dict_tgt[tgt_sent] = 1
#                     final_useful_lines.append(iNumLine)
#                 iNumLine += 1
#
#         del dict_tgt  # Save memory
#
#         for (size_file, size_svd) in dictSizesTrain.items():
#             if size_svd > len(final_useful_lines):
#                 print("ERROR: The size is higher than the size (%d) of the data (%d) for file (%s) and langs (%s-%s)"
#                       % (size_svd, len(final_useful_lines), file, src, tgt))
#                 return "NO"
#             final_size = len(final_useful_lines)  # Use them all
#             step = int(floor(len(final_useful_lines)/(NFOLDS*final_size)))
#             # if step >= 1:  # there are enough sentences for NFOLDS, if not then we need to do sampling with replacement
#             #     rand_numbers = final_useful_lines[:]
#             #     random.shuffle(rand_numbers)
#             # else:
#             #     rand_numbers = []
#             #     for nF in range(NFOLDS):
#             #         rand_numbers = rand_numbers + random.sample(final_useful_lines, final_size)
#
#             if step >= 1:  # there are enough sentences for NFOLDS, if not then we need to do sampling with replacement
#                 rand_numbers = final_useful_lines[:size_file*NFOLDS]
#                 # random.shuffle(rand_numbers)
#             else:
#                 rand_numbers = []
#                 for nF in range(NFOLDS):
#                     rand_numbers = rand_numbers + random.sample(final_useful_lines, final_size)
#
#             for nF in range(NFOLDS):
#                 rand_numbers_n = sorted(rand_numbers[nF*final_size:(nF+1)*final_size])
#                 train_file_out_path_tgt = train_file_path_tgt + '.' + tgt + '.' + str(size_file) + '.' + str(nF)
#                 t1 = datetime.datetime.now()
#                 if (not os.path.isfile(train_file_out_path_tgt)) or make_new is True:
#                     print("Creating " + train_file_out_path_tgt)
#
#                     # Create the new training files
#                     h_file_tgt = codecs.open(train_file_out_path_tgt, 'w', 'utf-8')
#
#                     added = 0
#                     numLine = 0
#                     with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
#                         for i in rand_numbers_n:
#                             while(numLine <= i):
#                                 tgt_sent = f.readline()
#                                 numLine += 1
#
#                             h_file_tgt.writelines(tgt_sent)
#                             added = added + 1
#                             if added > final_size:
#                                 break
#                     h_file_tgt.close()
#                     if added != final_size:
#                         print("ERROR: added only " + str(added) + "instead of " + str(final_size) + " for tgt: " +
#                               train_file_out_path_tgt)
#                         return "NO"
#
#                 t2 = datetime.datetime.now()
#                 print("Execution time: %s" % (t2-t1))
#
#         return "OK"
#     else:
#         print("ERROR: in " + file + " for files src: " + src + ' or tgt: ' + tgt + ' check that they exist')
#         return "NO"
#
#
# def createTrainingFiles(overwrite_all=False):
#     print("***** Creating training files ******")
#     pool = multiprocessing.Pool(processes=NUM_MAX_CORES)
#
#     tmpTask = []
#     for (lang_pair, files) in filesPerLanguage.items():
#         print('Processing language pair ' + lang_pair)
#         aPrefix = lang_pair.split("-")
#         src = aPrefix[0]
#         tgt = aPrefix[1]
#         for file in files:
#             tmpTask.append((train_data_dir + file, src, tgt, overwrite_all))
#
#     results = [pool.map(createtrainfile_wrapper, tmpTask)]
#     if len(results[0]) != len(tmpTask):
#         print("ERROR: Creating training files")
#         exit(-1)
#     print("... Done")
#     pool.close()
#     pool.join()
#     print("Finish")

######################################################################
class TrainingClass():
    def __init__(self, train_data_dir, filesPerLanguageForLM, filesPerLanguage, dictSizesTrain, cfg):

        self.train_data_dir = train_data_dir
        self.filesPerLanguageForLM = filesPerLanguageForLM
        self.filesPerLanguage = filesPerLanguage
        self.dictSizesTrain = dictSizesTrain
        self.cfg = cfg

    #PASO 1 crear los directorios de salida del entrenamiento

    def createOutputDirs(self):
        print("***** Creating output directories *****")

        if not os.path.exists(dir_lm_out):
            print("...creating " + dir_lm_out)
            os.makedirs(dir_lm_out)

        if not os.path.exists(dir_svd_mono):
            print("...creating " + dir_svd_mono)
            os.makedirs(dir_svd_mono)

        print("... Done")

   # def createLM_wrapper(self,args):
    #    return self.createLM(*args)

    # crea los modelos de lenguaje a partir de los ficheros que se van a entrenar sin preprocesar

    def fntTrainLMs(self, train_data_dir, filesPerLanguageForLM, overwrite_all=False):
        #    pool = MyPool(NUM_MAX_CORES)
        # number_of_workers = NUM_MAX_CORES
        number_of_cores = self.cfg['trainModule']['NUM_MAX_CORES']

        with Pool(number_of_cores) as p:
            tmpTask = []
            for (lang, files) in filesPerLanguageForLM.items():
                for file in files:
                    tmpTask.append((train_data_dir, lang, file, overwrite_all))

            # results = [p.map(createLM_wrapper, tmpTask)]
            results = [p.starmap(self.createLM, tmpTask)]
            for res in results[0]:
                if res == "NO":
                    print("ERROR: One of the LM could not be created.")
                    exit(-1)

        # pool.close()
        # pool.join()
        print("Finish")

    # Todo: Create a single text file with all the filenames or do it manually with all the training data
    def createLM(self, train_data_dir, lang, filename, bRetrain=False):
        # This function extract ngram counts for a given file. If the lm already exists the system skips its creation
        print("***** Extract n-gram counts for language " + lang + " from file: " + filename + " ******")

        txt_file = train_data_dir + '/' + filename + '.' + 'lower' + '.' + lang

        # if lang != 'en':  # We need first to tokenize in characters
        #     txt_file = train_data_dir + '/' + filename + '.' + lang + '.lower_' + lang
        #     with codecs.open(train_data_dir + '/' + filename + '.' + lang + '.lower', 'r', 'utf-8') as f, \
        #             codecs.open(train_data_dir + '/' + filename + '.' + lang + '.lower_' + lang, 'w', 'utf-8') as o:
        #         for line in f.readlines():
        #             n_l = ' '.join(list(line.strip()))
        #             o.write(n_l + '\n')

        MIN_NGRAM_ORDER = self.cfg['trainModule']['MIN_NGRAM_ORDER']
        MAX_NGRAM_ORDER = self.cfg['trainModule']['MAX_NGRAM_ORDER']

        path_file = os.path.dirname(filename)
        # Train language models
        if not os.path.exists(dir_lm_out + path_file + '/' + lang):
            print("...creating " + dir_lm_out + path_file + '/' + lang)
            os.makedirs(dir_lm_out + path_file + '/' + lang)

        for ngram_order in range(MIN_NGRAM_ORDER, MAX_NGRAM_ORDER + 1, 1):
            lm_file_out = dir_lm_out + path_file + '/' + lang + '/' + os.path.basename(
                filename) + '.' + lang + '.' + str(ngram_order) + '.lm'
            if not os.path.exists(lm_file_out) or bRetrain is True:
                if not os.path.exists(txt_file):
                    print('ERROR: file ' + txt_file + ' does not exists')
                    exit(-1)

                print("Creating n-gram counts for file %s and order %d" % (filename, ngram_order))
                cmd = 'ngram-count -unk -prune 1e-6 -minprune ' + str(ngram_order) + ' -order ' + str(
                    ngram_order) + ' -text ' + txt_file + ' -lm ' + lm_file_out
                print(cmd)
                p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                for line in p.stdout.readlines():
                    print('---- ' + filename + ' : ' + str(line))
                retval = p.wait()

        print("... Done " + lang + ' file: ' + filename)
        return "DONE"

    def monosvd_wrapper(self, args):
        return self.createMonolingualSVDforTaskParallel(*args)

    def createMonolingualSVDforTaskParallel(self, file, size_file, size_svd, nF, lang, retrain_svd):
        print("***** Creating Monolingual SVD for file " + file + ' (' + lang + ") and Fold " + str(nF) + " ******")
        t1 = datetime.datetime.now()
        print("Starting " + file + ' (' + lang + ") and Fold " + str(nF) + ' at ' + str(t1))
        # train_file_path = train_data_dir + file + '.lower.' + lang + '.' + str(size) + '.' + str(nF)
        if lang != 'en' and lang != 'ko' and lang != 'hi':
            train_file_path = self.train_data_dir + file + '.lower.' + lang + '.' + str(size_file) + '.' + str(nF)
        else:
            train_file_path = self.train_data_dir + file + '.lower.' + lang + '.' + str(size_file) + '.' + str(nF)
        # train_file_path = train_data_dir + file + '.lower'

        # svd_output_matrix = dir_svd_mono + '/' + file + '.' + lang + '.' + str(size) + '.' + str(nF) + '.mc' + str(MIN_COUNTS) # This is the full matrix without any reduction
        # This is the full matrix without any reduction
        svd_output_matrix = dir_svd_mono + '/' + file + '.' + lang + '.' + str(size_file) + '.' + str(
            size_svd) + '.' + str(nF)

        MIN_COUNTS = self.cfg['trainModule']['MIN_COUNTS']

        if os.path.isfile(train_file_path):
            if (not os.path.isfile(svd_output_matrix + '.h5') or not os.path.isfile(svd_output_matrix + '.dic')) \
                    or overwrite_all is True or retrain_svd is True:
                print('Reading training data from ' + train_file_path)
                train_data = codecs.open(train_file_path, 'r', 'utf-8')
                train_sentences = train_data.readlines()
                train_data.close()

                # Create the new training files
                vs = VectorSpace()
                # Trains the full matrix, later we will use the reduced dimensions
                vs.trainVectorSpace((size_svd), lang, MIN_COUNTS, train_sentences,
                                    transforms=type_vectorizer)  # Names of the output files
                vs.save(svd_output_matrix)
        else:
            print("ERROR: file " + train_file_path + ' does not exist.')
            return "NO"

        t2 = datetime.datetime.now()
        print("... Done " + file + ' (' + lang + ") and Fold " + str(nF) + ' in ' + str(t2 - t1))
        return "OK"

    #metodo para crear las matrices de SVD para el

    def createMonolingualSVDFromTrainingFiles(self, size_file, size_svd, nF, retrain_svd):
        print("***** Creating Monolingual SVD files for size " + str(size_svd) + " and Fold " + str(nF) + " ******")
        t1 = datetime.datetime.now()
        print("Starting at " + str(t1))

        num_cores_mono = {self.cfg['trainModule']['num_cores_mono']['Max']: self.cfg['trainModule']['num_cores_mono']['Min']}

        pool = multiprocessing.Pool(processes=num_cores_mono[size_svd])
        tmpTask = []
        for (lang_pair, files) in self.filesPerLanguage.items():
            print('Processing language pair ' + lang_pair)
            aPrefix = lang_pair.split("-")
            tgt = aPrefix[1]
            for file in files:
                tmpTask.append((file, size_file, size_svd, nF, tgt, retrain_svd))
                path_file = os.path.dirname(file)
                if not os.path.exists(dir_svd_mono + path_file):
                    print("...creating " + dir_svd_mono + path_file)
                    os.makedirs(dir_svd_mono + path_file)

        results = [pool.map(self.monosvd_wrapper, tmpTask)]

        for res in results[0]:
            if res == "NO":
                print("ERROR: One of the mono SVD matrices could not be created.")
                exit(-1)

        t2 = datetime.datetime.now()
        print("... Done size " + str(size_svd))
        print("Execution time: %s" % (t2 - t1))

        pool.close()
        pool.join()
        print("Finish")

    def fntCreateSVDs(self, args):
        global aTypeLingualExp

        dict_train_sizes = {self.cfg['runall']['dictSizesTrain']['Maxvalue']: self.cfg['runall']['dictSizesTrain']['Minvalue']}  # FOR WAT2018-My-En
        NFOLDS = self.cfg['trainModule']['NFOLDS']

        for typeLingualExp in aTypeLingualExp:
            print("*********************************************************************")
            print("*********************** " + typeLingualExp + " **********************")
            print("*********************************************************************")

            if typeLingualExp == "mono":
                for (train_size, svd_size) in dict_train_sizes.items():
                    for nF in range(NFOLDS):
                        self.createMonolingualSVDFromTrainingFiles( train_size, svd_size,
                                                              nF, args.ret_svd)

    def createtrainfile_wrapper(self,args):
        return self.createTrainingFilesParallel(*args)

    def createTrainingFilesParallel(self,file, src, tgt, make_new=False):
        print("***** Creating training file for " + file + ' src: ' + src + ' and tgt: ' + tgt + " ******")

        NFOLDS = self.cfg['trainModule']['NFOLDS']
        dictSizesTrain = {
            'MaxValue': self.cfg['trainModule']['dictSizesTrain']['Max'],
            'MinValue': self.cfg['trainModule']['dictSizesTrain']['Min']
        }

        if tgt != 'en' and tgt != 'ko' and tgt != 'hi':
            train_file_path_tgt = file + '.lower_' + tgt
        else:
            train_file_path_tgt = file + '.lower'

        if os.path.isfile(train_file_path_tgt):
            dict_tgt = {}
            iNumLine = 0
            final_useful_lines = []
            with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
                for tgt_sent in f:
                    if tgt_sent not in dict_tgt:
                        dict_tgt[tgt_sent] = 1
                        final_useful_lines.append(iNumLine)
                    iNumLine += 1

            del dict_tgt  # Save memory

            for (size_file, size_svd) in dictSizesTrain.items():
                if size_svd > len(final_useful_lines):
                    print(
                        "ERROR: The size is higher than the size (%d) of the data (%d) for file (%s) and langs (%s-%s)"
                        % (size_svd, len(final_useful_lines), file, src, tgt))
                    return "NO"
                final_size = len(final_useful_lines)  # Use them all
                step = int(floor(len(final_useful_lines) / (NFOLDS * final_size)))
                # if step >= 1:  # there are enough sentences for NFOLDS, if not then we need to do sampling with replacement
                #     rand_numbers = final_useful_lines[:]
                #     random.shuffle(rand_numbers)
                # else:
                #     rand_numbers = []
                #     for nF in range(NFOLDS):
                #         rand_numbers = rand_numbers + random.sample(final_useful_lines, final_size)

                if step >= 1:  # there are enough sentences for NFOLDS, if not then we need to do sampling with replacement
                    rand_numbers = final_useful_lines[:size_file * NFOLDS]
                    # random.shuffle(rand_numbers)
                else:
                    rand_numbers = []
                    for nF in range(NFOLDS):
                        rand_numbers = rand_numbers + random.sample(final_useful_lines, final_size)

                for nF in range(NFOLDS):
                    rand_numbers_n = sorted(rand_numbers[nF * final_size:(nF + 1) * final_size])
                    train_file_out_path_tgt = train_file_path_tgt + '.' + tgt + '.' + str(size_file) + '.' + str(nF)
                    t1 = datetime.datetime.now()
                    if (not os.path.isfile(train_file_out_path_tgt)) or make_new is True:
                        print("Creating " + train_file_out_path_tgt)

                        # Create the new training files
                        h_file_tgt = codecs.open(train_file_out_path_tgt, 'w', 'utf-8')

                        added = 0
                        numLine = 0
                        with codecs.open(train_file_path_tgt, 'r', 'utf-8') as f:
                            for i in rand_numbers_n:
                                while (numLine <= i):
                                    tgt_sent = f.readline()
                                    numLine += 1

                                h_file_tgt.writelines(tgt_sent)
                                added = added + 1
                                if added > final_size:
                                    break
                        h_file_tgt.close()
                        if added != final_size:
                            print("ERROR: added only " + str(added) + "instead of " + str(final_size) + " for tgt: " +
                                  train_file_out_path_tgt)
                            return "NO"

                    t2 = datetime.datetime.now()
                    print("Execution time: %s" % (t2 - t1))

            return "OK"
        else:
            print("ERROR: in " + file + " for files src: " + src + ' or tgt: ' + tgt + ' check that they exist')
            return "NO"

    def createTrainingFiles(self,overwrite_all=False):
        print("***** Creating training files ******")
        NUM_MAX_CORES = self.cfg['trainModule']['NUM_MAX_CORES']
        pool = multiprocessing.Pool(processes=NUM_MAX_CORES)

        tmpTask = []
        for (lang_pair, files) in filesPerLanguage.items():
            print('Processing language pair ' + lang_pair)
            aPrefix = lang_pair.split("-")
            src = aPrefix[0]
            tgt = aPrefix[1]
            for file in files:
                tmpTask.append((train_data_dir + file, src, tgt, overwrite_all))

        results = [pool.map(self.createtrainfile_wrapper, tmpTask)]
        if len(results[0]) != len(tmpTask):
            print("ERROR: Creating training files")
            exit(-1)
        print("... Done")
        pool.close()
        pool.join()
        print("Finish")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-overwrite', help='Overwrite all the files.', action='store_true')
    parser.add_argument('-ret_svd', help='Retrain the svd matrices.', action='store_true')

    parser.add_argument('-c', '--my-config', type=str, dest='MyConfigFilePath', required=False, help='config file path')
    parser.add_argument('-d', '--num_cores', dest='NofCores', help='Number of cores')  # this option can be set in a config file because it starts with '--'




    args = parser.parse_args()

    filepath = args.MyConfigFilePath

    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    FilesPerLanguageForLM = cfg['trainModule']['filesPerLanguageForLM']
    FilesPerLanguage = cfg['trainModule']['filesPerLanguage']

    overwrite_all = args.overwrite

    dictSizesTrain = {
        'Max': cfg['trainModule']['dictSizesTrain']['Max'],
        'Min': cfg['trainModule']['dictSizesTrain']['Min']
    }


    tClass= TrainingClass(train_data_dir, FilesPerLanguageForLM, FilesPerLanguage, dictSizesTrain, cfg)

    tClass.createOutputDirs()
    tClass.fntTrainLMs(train_data_dir, FilesPerLanguageForLM, overwrite_all=overwrite_all)
    tClass.createTrainingFiles(overwrite_all=overwrite_all)
    # tClass.fntCreateSVDs(train_data_dir, filesPerLanguage, dictSizesTrain, NFOLDS, args=args)

    tClass.fntCreateSVDs(args=args)


if __name__ == '__main__':
    main()











