# -*- coding: utf-8 -*-

import sys
# import cPickle as pickle
import pickle as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import re

try:
    from numpy import transpose
    from numpy import dot
    from numpy.linalg import norm
    from numpy import isnan
    from numpy import seterr
    from numpy import concatenate
    from numpy import zeros
    from numpy import average
    from numpy import diagonal
    from numpy import float32
except:
    print ("Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?")
    sys.exit()


class VectorSpace:
    """ A algebraic model for representing text documents as vectors of identifiers. 
    A document is represented as a vector. Each dimension of the vector corresponds to a 
    separate term. If a term occurs in the document, then the value in the vector is non-zero.
    """

    def __init__(self):
        self.vector_index_to_keyword_mapping = None

        self.vectorizer = None
        self.svd = None

    def trainVectorSpace(self, size_svd, lang, MIN_COUNTS, documents_src=[], documents_tgt=[], cross=False,
                         transforms='tfidf'):

        if cross is True:
            assert len(documents_tgt) > 0, "ERROR: the length of the target sentences is 0"

        assert len(documents_src) > 0, "ERROR: the length of the source sentences is 0"

        self._build(size_svd, lang, MIN_COUNTS, documents_src, documents_tgt, cross, transforms)

    def _build(self, size_svd, lang, MIN_COUNTS, documents_src, documents_tgt, cross, transform):
        """ Create the vector space for the passed document strings """

        if cross is True:
            if lang != 'en' and lang != 'ko' and lang != 'hi' and lang != 'in' and lang != 'my':
                texts_src = [' '.join([' '.join([c+'SRC' for c in list(word.strip())]) for word in document.split()])
                             for document in documents_src]
                texts_tgt = [' '.join([' '.join([c+'TGT' for c in list(word.strip())]) for word in document.split()])
                             for document in documents_tgt]
            else:
                texts_src = [' '.join([word + 'SRC' for word in document.split()]) for document in documents_src]
                texts_tgt = [' '.join([word + 'TGT' for word in document.split()]) for document in documents_tgt]
            vocab = self.fntCreateVocab(MIN_COUNTS, lang, texts_src, texts_tgt)
            texts = []
            for i in range(len(texts_src)):
                texts.append(texts_src[i] + ' ' + texts_tgt[i])
            del texts_src
            del texts_tgt
        else:
            if lang != 'en' and lang != 'ko' and lang != 'hi' and lang != 'in':
                texts = [' '.join([' '.join([c for c in list(word.strip())]) for word in document.split()]) for
                         document in documents_src]
            else:
                texts = [' '.join([word for word in document.split()]) for document in documents_src]
            vocab = self.fntCreateVocab(MIN_COUNTS, lang, texts)

        print('transform: ' + transform)
        if transform == 'tfidf':
            self.vectorizer = TfidfVectorizer(lowercase=False, min_df=MIN_COUNTS, use_idf=True, smooth_idf=True,
                                              token_pattern=r"(?u)\b\w+\b", vocabulary=vocab, dtype=float32)
        elif transform == 'binary':
            self.vectorizer = TfidfVectorizer(lowercase=False, min_df=MIN_COUNTS, use_idf=True, smooth_idf=True,
                                              binary=True, norm=u'l2', token_pattern=r"(?u)\b\w+\b", vocabulary=vocab,
                                              dtype=float32)
        elif transform == 'all_binary':
            self.vectorizer = TfidfVectorizer(lowercase=False, min_df=MIN_COUNTS, use_idf=False, smooth_idf=False,
                                              binary=True, norm=u'l2', token_pattern=r"(?u)\b\w+\b", vocabulary=vocab)
        else: # Normal counts
            self.vectorizer = CountVectorizer(lowercase=True, min_df=0, vocabulary=vocab, token_pattern=r"(?u)\b\w+\b")

        X = self.vectorizer.fit_transform(texts)
        #size_svd = min(X.shape[0], X.shape[1], size_svd) - 1
        size_svd = min(X.shape[0], X.shape[1], size_svd)

        print(str(size_svd))
        # self.svd = TruncatedSVD(size_svd, 'arpack')
        # del texts
        # self.svd.fit(X)
        # print('x')
        import numpy as np
        u,s,v = np.linalg.svd(X.todense().T)
        self.svd = u[:, 0:size_svd]


    def fntCreateVocab(self, MIN_COUNTS, lang, src, tgt=[]):
        vocab = {}
        offset = 0
        if lang != 'en' and lang != 'ko' and lang != 'hi' and lang != 'in' and lang != 'my':
            tokens = [[' '.join([c for c in list(item.strip())]) for item in document.split()] for document in src]
        else:
            tokens = [[item for item in document.split()] for document in src]

        all_tokens = sum(tokens, [])
        tokens_pass = set(word for word in set(all_tokens) if all_tokens.count(word) >= MIN_COUNTS )

        # Associate a position with the keywords which maps to the dimension on the vector used to represent this word
        unique_vocabulary_list = set(tokens_pass)
        for word in unique_vocabulary_list:
            if re.search(r'(?u)(\b\w+\b)', word):
                vocab[word] = offset
                offset += 1

        # If we are passing tgt then it is a cross lingual setup
        if len(tgt) > 0:
            if lang != 'en' and lang != 'ko' and lang != 'hi' and lang != 'in' and lang != 'my':
                tokens = [[' '.join([c for c in list(item.strip())]) for item in document.split()] for document in tgt]
            else:
                tokens = [[item for item in document.split()] for document in tgt]
            all_tokens = sum(tokens, [])
            tokens_pass = set(word for word in set(all_tokens) if all_tokens.count(word) >= MIN_COUNTS )
            unique_vocabulary_list = set(tokens_pass)
            for word in unique_vocabulary_list:
                if re.search(r'(?u)(\b\w+\b)', word):
                    vocab[word] = offset
                    offset += 1

        return vocab

    def search(self, lang, ref_src_sentences, translated_sentences, size_svd, cross=False):
        """ search for documents that match based on a list of terms """

        assert len(ref_src_sentences) == len(translated_sentences), "ERROR: the length of the source/reference (%d) and " \
                                                                    "target (%d) sentences are not the same" % \
                                                                    (len(ref_src_sentences), len(translated_sentences))

        if lang != 'en' and lang != 'ko' and lang != 'hi' and lang != 'in' and lang != 'my':
            if cross is True:
                texts = [' '.join([' '.join([c+'SRC' for c in list(word.strip())]) for word in document.split()]) for
                         document in ref_src_sentences]
                reference_vector = self.vectorizer.transform(texts)
                del texts

                texts = [' '.join([' '.join([c+'TGT' for c in list(word.strip())]) for word in document.split()]) for
                         document in translated_sentences]
                target_vector = self.vectorizer.transform(texts)
                del texts
            else:
                texts = [' '.join([' '.join([c for c in list(word.strip())]) for word in document.split()]) for
                         document in ref_src_sentences]
                reference_vector = self.vectorizer.transform(texts)
                del texts

                texts = [' '.join([' '.join([c for c in list(word.strip())]) for word in document.split()]) for
                         document in translated_sentences]
                target_vector = self.vectorizer.transform(texts)
                del texts
        else:
            if cross is True:
                texts = [' '.join([word + 'SRC' for word in document.split()]) for document in ref_src_sentences]
                reference_vector = self.vectorizer.transform(texts)
                del texts

                texts = [' '.join([word + 'TGT' for word in document.split()]) for document in translated_sentences]
                target_vector = self.vectorizer.transform(texts)
                del texts
            else:
                texts = [' '.join([word for word in document.split()]) for document in ref_src_sentences]
                reference_vector = self.vectorizer.transform(texts)
                del texts

                texts = [' '.join([word for word in document.split()]) for document in translated_sentences]
                target_vector = self.vectorizer.transform(texts)
                del texts

        #svd_components = self.svd.components_[0:size_svd, :]
        svd_components = self.svd[:, 0:size_svd]
        cosines = self._cosine(target_vector, reference_vector, svd_components)
        # cosines[cosines<1e-9]=0
        return cosines


    def _cosine(self, target, reference, svd_components):
        """ related documents j and q are in the concept space by comparing the vectors :
            cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """

        tgt = sparse.csr_matrix.dot(target, svd_components)
        # tgt[tgt<0]=0
        ref = sparse.csr_matrix.dot(reference, svd_components)
        # ref[ref<0]=0
        # return diagonal(cosine_similarity(ref, tgt))
        #return cosine_similarity(ref, tgt)  # Return everything, i.e. the NxN matrix of all possible cosine distances
        r = cosine_similarity(ref, tgt)
        r[r<0]=0.0
        return r

    def save(self, name_model):
        outFile = open(name_model + '.dic', "wb")
        pickle.dump(self.vectorizer, outFile, -1)
        outFile.close()
        joblib.dump(self.svd, name_model + '.h5', compress=7)

    def load(self, name_model):
        self.svd = joblib.load(name_model + '.h5')
        file_h = open(name_model + '.dic', "rb")
        self.vectorizer = pickle.load(file_h)
        file_h.close()



