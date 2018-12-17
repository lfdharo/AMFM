#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
__author__ = 'luisdhe'
__version__ = "$Revision: 1.0.4 $"

# Common python modules
import os, sys, string
import argparse
import codecs
import numpy as np
import json
from socket import *

host = 'localhost'  # '127.0.0.1' can also be used
DEFAULT_PORT = 52000

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


    sock = socket()
    # Connecting to socket
    sock.connect((host, DEFAULT_PORT))  # Connect takes tuple of host and port


    # Example for calculating scores on English as target language
    # Calculate for each submmited sentence, given the reference, the AM, FM and combined scores
    for (target, submission) in zip(target_sentences_en, submission_sentences_en):
        (res_am, res_fm, res_am_fm) = fntSendDataGetResults(target, submission, 'en', sock, am=True, fm=True)
        print (res_fm, res_am, res_am_fm)

    # Example for calculating scores on Japanese as target language
    # Calculate for each submitted sentence, given the reference, the AM, FM and combined scores
    for (target, submission) in zip(target_sentences_jp, submission_sentences_jp):
        (res_am, res_fm, res_am_fm) = fntSendDataGetResults(target, submission, 'en', sock, am=True, fm=True)
        print (res_fm, res_am, res_am_fm)


def fntSendDataGetResults(target, submission, lang, sock, am, fm):
    message = dict()
    message['ref'] = target
    message['out'] = submission
    message['lang'] = lang
    message['am'] = am
    message['fm'] = fm
    message['data'] = ''
    res_message = sock.send(json.dumps(message))
    while True:
        res_message = sock.recv(16384)
        if res_message != '':
            break

    res_message = json.loads(res_message)
    if 'data' in res_message and res_message['data'] == 'finish':
        print ('The server sends an error message (%s) and finished the connection.' % (res_message['err_msg']))
        return None

    res_fm = -1.0
    if fm is True:
        res_fm = res_message['fm']

    res_am = -1.0
    if am is True:
        res_am = res_message['am']

    alpha = 0.5
    if 'alpha' in res_message:
        alpha = res_message['alpha']

    res_am_fm = -1.0
    if am is True and fm is True:
        res_am_fm = res_message['am_fm']

    return (res_am, res_fm, res_am_fm, alpha)


def processSubmission(ref, output, sock, am=True, fm=True, lang='en'):
    print('\n********* START PROCESSING SUBMISSION ************')
    print('Processing submissions ref=%s and output=%s' % (ref, output))
    try:
        target_sentences = []
        with codecs.open(ref, 'r', 'utf-8') as f_in:
            for line in f_in.readlines():
                target_sentences.append(line.strip())

        submission_sentences = []
        with codecs.open(output, 'r', 'utf-8') as f_in:
            for line in f_in.readlines():
                submission_sentences.append(line.strip())

        if len(target_sentences) != len(submission_sentences):
            print("******* ERROR: sentence lengths are not the same for files %s (%d) and %s (%d)" % (ref, len(target_sentences), output, len(submission_sentences)))
            raise error
        
        # Calculate for each submitted sentence, given the reference, the AM, FM and combined scores
        results = []
        print ('N_SENT,\tFM,\tAM,\tAM_FM')
        for num, (target, submission) in enumerate(zip(target_sentences, submission_sentences)):
            res = fntSendDataGetResults(target, submission, lang, sock, am, fm)
            if res is None:
                return
            else:
                (res_am, res_fm, res_am_fm, alpha) = res
            print ('%d,\t%.5f,\t%.5f,\t%.5f  ' % (num+1, res_fm, res_am, res_am_fm))
            results.append((res_fm, res_am, res_am_fm))

        # Calculate FM global score
        if fm is True:
            print ('GLOBAL AVERAGE FM: %.5f' % np.average([r[0] for r in results]))

        # Calculate AM global score
        if am is True:
            print ('GLOBAL AVERAGE AM: %.5f' % np.average([r[1] for r in results]))

        # Calculate Interpolated AM_FM score
        if am is True and fm is True:
            print ('GLOBAL AVERAGE AM_FM (%.2f): %.5f' % (alpha, np.average([r[2] for r in results])))

        print('********* END PROCESSING SUBMISSION ************\n')
        msg = dict()
        msg['data'] = 'finish'
        sock.send(json.dumps(msg))
    except:
        print ('ERROR: Skipping submissions ref=%s and output=%s' % (ref, output))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", help="file with the gold standard sentences")
    parser.add_argument("out", help="file with the submission sentences")
    parser.add_argument("lang", help="target language [en|jp]")
    parser.add_argument("-port", "--port", default=DEFAULT_PORT, help="port used to connect to the server")
    parser.add_argument("-list", "--list", action="store_true", help="ref and out files contains lists of parallel "
                                                                     "submission files to process. This speeds up the "
                                                                     "process since models are loaded only one time."
                                                                     "Submissions must be for the same language")
    parser.add_argument("-fm", "--fm", help="Do not calculate FM score", action="store_false")
    parser.add_argument("-am", "--am", help="Do not calculate AM score", action="store_false")

    args = parser.parse_args()

    sock = socket()
    # Connecting to socket
    sock.connect((host, int(args.port)))  # Connect takes tuple of host and port
    print('Connected to server on %s:port %s' %(host, args.port))

    if args.list is True:
        list_ref = []
        with codecs.open(args.ref, 'r', 'utf-8') as f_in:
            for line in f_in.readlines():
                list_ref.append(line.strip())

        list_out = []
        with codecs.open(args.out, 'r', 'utf-8') as f_in:
            for line in f_in.readlines():
                list_out.append(line.strip())

        assert len(list_ref) == len(list_out), "******* ERROR: number of submissions and references are not the same " \
                                                "for files %s (%d) and %s (%d)" % (args.ref,
                                                                  len(list_ref),
                                                                  args.out,
                                                                  len(list_out))

        for (ref, out) in zip(list_ref, list_out):
            processSubmission(ref, out, sock, am=args.am, fm=args.fm, lang=args.lang)
    else:
        processSubmission(args.ref, args.out, sock, am=args.am, fm=args.fm, lang=args.lang)


if __name__ == '__main__':
    main()