__author__ = 'luisdhe'

import os
import argparse
import configargparse as configuration
import yaml

import preProcessModule as pp
import trainModule as tm
import calculateScores as cs

root_dir = os.path.dirname(os.path.realpath(__file__)) #con esta sentencia estamos partiedno el origen desde el Train para bajo
submissions_dir = root_dir + '/'
train_data_dir = root_dir + '/'
scripts_dir = root_dir + '/tools/'

# Relative path for files to be used for training the language models
#
filesPerLanguageForLM = {
    # 'en': [
    #     'ASPEC/ASPEC-JE/train/train',
    # ],
    # 'jp': [
    #     'ASPEC/ASPEC-JE/train/train',
    # ],
    # 'zh': [
    #     'ASPEC/ASPEC-JC/train/train',
    # ],
    # 'jp': [
    #     'ASPEC/ASPEC-JC/train/train',
    # ],

    # 'en': [
    #     'JPO_PATENT_CORPUS/EJ/TRAIN/train',
    # ],
    # 'jp': [
    #     'JPO_PATENT_CORPUS/EJ/TRAIN/train',
    # ],

    # 'ko': [
    #     'JPO_PATENT_CORPUS/KJ/TRAIN/train',
    # ],
    # 'jp': [
    #     'JPO_PATENT_CORPUS/KJ/TRAIN/train',
    # ],
    #
    # 'zh': [
    #     'JPO_PATENT_CORPUS/CJ/TRAIN/train',
    # ],
    # 'jp': [
    #     'JPO_PATENT_CORPUS/CJ/TRAIN/train',
    # ],
    #
    # 'in': [
    #     'BPPT_CORPUS/train',
    # ],
    # 'en': [
    #     'BPPT_CORPUS/train',
    # ],

    # 'hi': [
    #     'Hindi/parallel/IITB.en-hi',
    # ],
    # 'en': [
    #     'Hindi/parallel/IITB.en-hi',
    # ],

    # 'jp': [
    #     'WAT2016-Ja-Hi/train',
    # ],
    # 'hi': [
    #     'WAT2016-Ja-Hi/train',
    # ],

    'jp': [
        'JIJI_CORPUS/train',
    ],
    'en': [
        'JIJI_CORPUS/train',
    ],

    # 'jp': [
    #     'RecipeCorpus_JaEn/train_titles',
    #     'RecipeCorpus_JaEn/train_steps',
    #     'RecipeCorpus_JaEn/train_ingredients',
    #     'RecipeCorpus_JaEn/dev_titles',
    #     'RecipeCorpus_JaEn/dev_steps',
    #     'RecipeCorpus_JaEn/dev_ingredients',
    #     'RecipeCorpus_JaEn/devtest_titles',
    #     'RecipeCorpus_JaEn/devtest_steps',
    #     'RecipeCorpus_JaEn/devtest_ingredients',
    #     'RecipeCorpus_JaEn/train_all',
    # ],
    # 'en': [
    #     'RecipeCorpus_JaEn/train_titles',
    #     'RecipeCorpus_JaEn/train_steps',
    #     'RecipeCorpus_JaEn/train_ingredients',
    #     'RecipeCorpus_JaEn/dev_titles',
    #     'RecipeCorpus_JaEn/dev_steps',
    #     'RecipeCorpus_JaEn/dev_ingredients',
    #     'RecipeCorpus_JaEn/devtest_titles',
    #     'RecipeCorpus_JaEn/devtest_steps',
    #     'RecipeCorpus_JaEn/devtest_ingredients',
    #     'RecipeCorpus_JaEn/train_all',
    # ],

    # 'my': [
    #     'wat2018.my-en/alt/train.alt',
    #     'wat2018.my-en/alt/dev.alt',
    #     'wat2018.my-en/alt/test.alt',
    # ],
    # 'en': [
    #     'wat2018.my-en/alt/train.alt',
    #     'wat2018.my-en/alt/dev.alt',
    #     'wat2018.my-en/alt/test.alt',
    # ],
}

filesPerLanguage = {
    # 'en-jp': [
    #     'ASPEC/ASPEC-JE/train/train-1',
    # ],
    # 'jp-en': [
    #     'ASPEC/ASPEC-JE/train/train-1',
    # ],
    # 'jp-zh': [
    #     'ASPEC/ASPEC-JC/train/train',
    # ],
    # 'zh-jp': [
    #     'ASPEC/ASPEC-JC/train/train',
    # ],

    # 'en-jp': [
    #     'JPO_PATENT_CORPUS/EJ/TRAIN/train',
    # ],
    # 'jp-en': [
    #     'JPO_PATENT_CORPUS/EJ/TRAIN/train',
    # ],

    # 'ko-jp': [
    #     'JPO_PATENT_CORPUS/KJ/TRAIN/train',
    # ],
    # 'jp-ko': [
    #     'JPO_PATENT_CORPUS/KJ/TRAIN/train',
    # ],

    # 'zh-jp': [
    #     'JPO_PATENT_CORPUS/CJ/TRAIN/train',
    # ],
    # 'jp-zh': [
    #     'JPO_PATENT_CORPUS/CJ/TRAIN/train',
    # ],
    #
    # 'in-en': [
    #     'BPPT_CORPUS/train',
    # ],
    # 'en-in': [
    #     'BPPT_CORPUS/train',
    # ],
    # 'hi-en': [
    #     'Hindi/parallel/IITB.en-hi',
    # ],
    # 'en-hi': [
    #     'Hindi/parallel/IITB.en-hi',
    # ],

    # 'jp-hi': [
    #     'WAT2016-Ja-Hi/train',
    # ],
    # 'hi-jp': [
    #     'WAT2016-Ja-Hi/train',
    # ],
    'jp-en': [
        'JIJI_CORPUS/train',
    ],
    'en-jp': [
        'JIJI_CORPUS/train',
    ],
    # 'jp-en': [
    #     # 'RecipeCorpus_JaEn/train_titles',
    #     # 'RecipeCorpus_JaEn/train_steps',
    #     # 'RecipeCorpus_JaEn/train_ingredients',
    #     'RecipeCorpus_JaEn/train_all',
    # ],
    # 'en-jp': [
    #     # 'RecipeCorpus_JaEn/train_titles',
    #     # 'RecipeCorpus_JaEn/train_steps',
    #     # 'RecipeCorpus_JaEn/train_ingredients',
    #     'RecipeCorpus_JaEn/train_all',
    # ],

    # 'my-en': [
    #     'wat2018.my-en/alt/train.alt',
    #     'wat2018.my-en/alt/dev.alt',
    # ],
    # 'en-my': [
    #     'wat2018.my-en/alt/train.alt',
    #     'wat2018.my-en/alt/dev.alt',
    # ],

}

submissionsPerLanguagePerYear = {
    # 'en-jp':{
    #     '2015':{
    #        'source': 'ASPEC/ASPEC-JE/devtest/devtest.en',
    #        'reference': 'ASPEC/ASPEC-JE/devtest/devtest.jp.ref',
    #        'test_id': 'devtest_en-jp',
    #        'submissions': [
    #            'ASPEC/ASPEC-JE/devtest/devtest.jp.google',
    #        ],
    #     },
    # },
    # 'jp-en':{
    #     '2015':{
    #        'source': 'ASPEC/ASPEC-JE/devtest/devtest.jp',
    #        'reference': 'ASPEC/ASPEC-JE/devtest/devtest.en.ref',
    #        'test_id': 'devtest_jp-en',
    #        'submissions': [
    #            'ASPEC/ASPEC-JE/devtest/devtest.en.google',
    #        ],
    #     },
    # },
    # 'en-jp': {
    #     '2015': {
    #         'source': 'ASPEC/ASPEC-JE/test/test.en',
    #         'reference': 'ASPEC/ASPEC-JE/test/test.jp',
    #         'test_id': 'test_en-jp.bl_hierM',
    #         'submissions': [
    #             'ASPEC/ASPEC-JE/submissions/bl_output/aspec_je/hierModel/test_en2ja.out',
    #         ],
    #     },
    # },
    # 'jp-en': {
    #     '2015': {
    #         'source': 'ASPEC/ASPEC-JE/test/test.jp',
    #         'reference': 'ASPEC/ASPEC-JE/test/test.en',
    #         'test_id': 'test_jp-en.bl_hierM',
    #         'submissions': [
    #             'ASPEC/ASPEC-JE/submissions/bl_output/aspec_je/fake_hierModel/test_ja2en.out',
    #         ],
    #     },
    # },
    # 'zh-ja': {
    #     '2015': {
    #         'source': 'ASPEC/ASPEC-JC/test/test.zh',
    #         'reference': 'ASPEC/ASPEC-JC/test/test.ja.lower',
    #         'test_id': 'test_ja-zh.bl_hierM',
    #         'submissions': [
    #             'ASPEC/ASPEC-JC/submissions/bl_output/CJ/hierModel/test_zh2ja.out',
    #         ],
    #     },
    # },

    # 'en-jp': {
    #     'JPO_2015': {
    #         'source': 'JPO_PATENT_CORPUS/EJ/DEVTEST/devtest.en',
    #         'reference': 'JPO_PATENT_CORPUS/EJ/DEVTEST/devtest.jp',
    #         'test_id': 'devtest_jp',
    #         'submissions': [
    #             'JPO_PATENT_CORPUS/EJ/DEVTEST/devtest.jp',
    #         ],
    #     },
    # },

    # 'ko-jp': {
    #     'JPO_2015': {
    #         'source': 'JPO_PATENT_CORPUS/KJ/DEVTEST/devtest.jp',
    #         'reference': 'JPO_PATENT_CORPUS/KJ/DEVTEST/devtest_reduced.ko',
    #         'test_id': 'devtest_ko',
    #         'submissions': [
    #             'JPO_PATENT_CORPUS/KJ/DEVTEST/devtest_test.ko',
    #         ],
    #     },
    # },
}

#NFOLDS = 1


dictSizesTrain = {15000: 2500}  # FOR WAT2018-My-En
# dictSizesTrain = {13000: 2500}  # FOR WAT2016-Ja-Hi
# dictSizesTrain = {10000: 2500}
# dictSizesTrain = {7000: 1500} # FOR HINDEN_HI_EN


def main():

    # config = configuration.ArgParser(default_config_files=['/home/enrique/Escritorio/TFG_Pendrive/AMFM/Settings.yaml'])
    # config = configuration.ArgParser()
    # config.add('-c', '--my-config', type=str, dest='MyConfigFilePath', required=False, is_config_file=False, help='config file path')
    # config.add('--genome', required=True,help='path to genome file')  # this option can be set in a config file because it starts with '--'
    # config.add('-d', '--num_cores',dest='NofCores', help='Number of cores',env_var='NUM_CORES')  # this option can be set in a config file because it starts with '--'
    # args1 = config.parse_args()
    # numcores = args1.NofCores
    # filepath = args1.MyConfigFilePath
    # print(numcores)
    # print(filepath)




    parser = argparse.ArgumentParser()
    parser.add_argument('-overwrite', help='Overwrite all the files.', action='store_true')
    parser.add_argument('-ret_svd', help='Retrain the svd matrices.', action='store_true')
    parser.add_argument('-cross', help='Do only cross.', action='store_true')
    parser.add_argument('-mono', help='Do only mono.', action='store_true')
    parser.add_argument('-average', help='Average the results from different SVD files.', action='store_true')
    parser.add_argument('-c', '--my-config', type=str, dest='MyConfigFilePath', required=False, is_config_file=False, help='config file path')
    parser.add_argument('-d', '--num_cores', dest='NofCores', help='Number of cores', env_var='NUM_CORES')  # this option can be set in a config file because it starts with '--'

    args = parser.parse_args()
    numcores = args.NofCores
    filepath = args.MyConfigFilePath
    print('The number of Cores you setted up to use are: ' + numcores)
    print('The current configuration file you are using is : ' + filepath)

    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # for element in cfg:
    #     if element["calculatescores"] == "calculatescores":
    #         for subelement in element:
    #             if subelement["name"] == "NFOLDS":
    #                 subelement["value"] = "4"

    NFOLDS = cfg['runall']['NFOLDS']
    dictSizesTrain = {cfg['runall']['dictSizesTrain']['Maxvalue']: cfg['runall']['dictSizesTrain']['Minvalue']}  # FOR WAT2018-My-En

    overwrite_all = args.overwrite
    # overwrite_all = True
    # Preprocessing
    cP = pp.PreProcessingClass(submissionsPerLanguagePerYear=submissionsPerLanguagePerYear,
                              filesPerLanguage=filesPerLanguage, filesPerLanguageForLM=filesPerLanguageForLM,
                              train_data_dir=train_data_dir, scripts_dir=scripts_dir, submissions_dir=submissions_dir, cfg=cfg)
    cP.preprocess_files(bDoAll=overwrite_all)

    # Create the training set for the SVD
    # pp.createTrainingFiles(filesPerLanguage, overwrite_all=overwrite_all) #original
    cP.createTrainingFiles(filesPerLanguage, overwrite_all=overwrite_all)
    # pp.PreProcessingClass.createTrainingFiles(filesPerLanguage, overwrite_all=overwrite_all)
    #raw_input('Press a key to continue')

    # Training SVD and creating training files
    # tm.createOutputDirs()
    # tm.fntTrainLMs(train_data_dir, filesPerLanguageForLM, overwrite_all=overwrite_all)
    # tm.fntCreateSVDs(train_data_dir, filesPerLanguage, dictSizesTrain, NFOLDS, args=args)

    tM = tm.TrainingClass(train_data_dir, filesPerLanguageForLM, filesPerLanguage, dictSizesTrain, cfg)
    tM.createOutputDirs()
    tM.fntTrainLMs(train_data_dir, filesPerLanguageForLM, overwrite_all=overwrite_all) #ayuda para dejar esto clean
    tM.fntCreateSVDs(args=args)
    # Calculating scores
    # cs.fntProcessSubmissions(submissionsPerLanguagePerYear, overwrite_all=overwrite_all, bAverageFiles=False, fpl=filesPerLanguage)


if __name__ == '__main__':
    main()