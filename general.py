__author__ = 'luisdhe'
##import cPickle as pickle
import pickle as pickle

def load_lm_model(lm_model_path):
    print ("***** Load lm metrics model *****")
    file_h = open(lm_model_path, "rb")
    lm_scores_per_sentence_allorders = pickle.load(file_h)
    file_h.close()
    print ("... Done")
    return lm_scores_per_sentence_allorders

def load_lingual_model(lingual_model_path):
    print ("***** Load lingual metrics model *****")
    file_h = open(lingual_model_path, "rb")
    scores_allsvdsizes = pickle.load(file_h)
    file_h.close()
    print ("... Done")
    return (scores_allsvdsizes)


def load_am_model(am_model_path):
    print ("***** Load am metrics model *****")
    file_h = open(am_model_path, "rb")
    scores_allsvdsizes = pickle.load(file_h)
    file_h.close()
    print ("... Done")
    return scores_allsvdsizes
