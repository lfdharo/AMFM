
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
#import codecs
import numpy as np


sc = set(['-', "'", '%'])
to_remove = ''.join([c for c in string.punctuation if c not in sc])
table = dict((ord(char), u'') for char in to_remove)


sc = set([',', '!', '?', '.'])
to_separate = ''.join([c for c in string.punctuation if c not in sc])
table_separate = dict((ord(char), u' ' + char) for char in to_separate)

# Examples
def demo():
    source_sentences_jp = ['本結果は両薬物の臨床効果と必ずしも一致せず，行動薬理学的指標として不十分だと示唆した。',
                           'さらに６年前にエナメル上皮腫再発のため顎下腺を含めた腫よう切除術を受けた。']
    target_sentences_en = ['These results suggest that fore-paw treading is not a reliable behavioural pharmacological '
                           'parameter for accessing the clinical effects of paroxetine or tianeptine, and that repeated '
                           'administration of tianeptine may enhance 5-HT1A receptor functioning.',
                           'The ameloblastoma recurred in the left submandibular region and was extirpated with the left '
                           'submandibular gland in 1995.']
    submission_sentences_en = ['The results are not necessarily consistent with the clinical efficacy of both drugs, '
                               'suggesting that it is insufficient as a behavioral pharmacology indicators.',
                               'Further he received a tumor resection, including the submandibular gland for ameloblastoma'
                               ' recurrence six years ago.']

    source_sentences = ['These results suggest that fore-paw treading is not a reliable behavioural pharmacological '
                        'parameter for accessing the clinical effects of paroxetine or tianeptine, and that repeated '
                        'administration of tianeptine may enhance 5-HT1A receptor functioning.',
                        'The ameloblastoma recurred in the left submandibular region and was extirpated with the left '
                        'submandibular gland in 1995.']
    target_sentences_jp = ['本結果は両薬物の臨床効果と必ずしも一致せず，行動薬理学的指標として不十分だと示唆した。',
                           'さらに６年前にエナメル上皮腫再発のため顎下腺を含めた腫よう切除術を受けた。']
    submission_sentences_jp = ['これらの結果は、フォア足の踏み込みがパロキセチンまたはチアネプチンの臨床効果にアクセスするための信頼性の'
                               '高い行動薬理学的なパラメータではないことを示唆している、とチアネプチンの反復投与は、5-HT1A受容体の機能'
                               'を高めることができます。',
                               'エナメル上皮腫は、左顎下の領域に再発し、1995年に左顎下腺で摘出しました。']

    #sock = socket()
    # Connecting to socket
    #sock.connect((host, DEFAULT_PORT))  # Connect takes tuple of host and port

    # Example for calculating scores on English as target language
    # Calculate for each submmited sentence, given the reference, the AM, FM and combined scores
    for (target, submission) in zip(target_sentences_en, submission_sentences_en):
        (res_am, res_fm, res_am_fm) = fntSendDataGetResults(target, submission, 'en', sock, am=True, fm=True)
        print(res_fm, res_am, res_am_fm)

    # Example for calculating scores on Japanese as target language
    # Calculate for each submitted sentence, given the reference, the AM, FM and combined scores
    for (target, submission) in zip(target_sentences_jp, submission_sentences_jp):
        (res_am, res_fm, res_am_fm) = fntSendDataGetResults(target, submission, 'en', sock, am=True, fm=True)
        print(res_fm, res_am, res_am_fm)

##### Metodo inicial que comienza to do lo que tiene que hacer el calc AMFM

def processSubmissionNew(target, submission, cs, fm, am):
    ### if de que si target y submission ya estan preprocesados no hacer nada de preprocesado
    (target, submission) = cs.doProcessFromStrings(ref=target, pred=submission)
    results = []
    alpha = cs.alpha
    with open(target,'r') as r1, open(submission,'r') as r2:
        for line_target,line_submission in zip(r1,r2):

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
        print('%i,\t%.5f,\t%.5f,\t%.5f  ' % (num+1, line[0], line[1], line[2]) )
        #(num+1, results(num+1), results(numa+2), results(num+3))

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

    # Perform basic pre-processing applied during training
    def doProcessFromStrings(self, ref, pred):
        ref = self.preProcess(ref, self.lang)
        pred = self.preProcess(pred, self.lang)
        return ref, pred

    # Pre-Processing for each sentence. In the case of languages different to English we perform tokenization
    # per character
    def preProcess(self, s, lang):

        # Creates the specfic File in "Files_input" for storing all the out put files related to each Language
        if not os.path.exists('preProcessed'):
            print("...creating " + 'preProcessed')
            os.makedirs('preProcessed')

        decompositionOfThePath = s.split('/')
        filename = decompositionOfThePath[len(decompositionOfThePath)-1] #we extract the filename

        with open(s, 'r') as f_in, open(self.TestDir+'/preProcessed/'+filename, 'w+') as f_out:
            for line in f_in:  # f_in.readlines():
                if len(line) == 0:  # To avoid empty lines
                    return '_EMPTY_'
                # Remove some punctuation
                s = line.translate(table)
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
                f_out.write(s + '\n')
        return f_out.name

    # Function to calculate the FM metric using language models
    def calculateFMMetric(self, ref, tst):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="file with the gold standard sentences")
    parser.add_argument("out", help="file with the submission sentences")
    parser.add_argument("lang", help="target language [en|jp]")
    ##esto podria estar to do contenido dentro del CFG (abajo)
  #  parser.add_argument("dataset", help="Dataset prefix [" + ', '.join(valid_datasets) + ']')
  #  parser.add_argument("-port", "--port", default=DEFAULT_PORT, help="port used to connect to the server")
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
    filepath = args.MyConfigFilePath    #path for the configuration File is input when executing script with -c argument

    if filepath == '' or filepath == None:
        print('You have not provided a configuration file')
        sys.exit()
    else:
        print('The current configuration file you are using is : ' + filepath)
        print('The number of Cores you setted up to use are: ' + str(numcores))


    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

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
            processSubmissionNew(ref, out,cs, am=args.am, fm=args.fm)
    else:
        processSubmissionNew(args.ref, args.out, cs, am=args.am, fm=args.fm)


if __name__ == '__main__':
    main()