__author__ = 'luisdhe' , 'vanmaren'

import os
import argparse
import configargparse as configuration
import yaml
import sys

import preProcessModule as pp
import trainModule as tm
# import calculateScores as cs



def main():


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
    filepath = args.MyConfigFilePath    #path for the configuration File is input when executing script with -c argument

    if filepath == '' or filepath == None:
        print('You have not provided a configuration file')
        sys.exit()
    else:
        print('The current configuration file you are using is : ' + filepath)
        print('The number of Cores you setted up to use are: ' + str(numcores))


    with open(filepath, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    dictSizesTrain = {
        'MaxValue': cfg['runall']['dictSizesTrain']['Maxvalue'],
        'MinValue': cfg['runall']['dictSizesTrain']['Minvalue']
    }  # FOR WAT2018-My-En

    submissionsPerLanguagePerYear = cfg['runall']['submissionsPerLanguagePerYear']
    filesPerLanguageForLM = cfg['runall']['filesPerLanguageForLM']
    filesPerLanguage = cfg['runall']['filesPerLanguage']
    overwrite_all = args.overwrite
    overwrite_all = True


    # Preprocessing

    cP = pp.PreProcessingClass(submissionsPerLanguagePerYear=submissionsPerLanguagePerYear,
                               filesPerLanguage=filesPerLanguage,
                               filesPerLanguageForLM=filesPerLanguageForLM,
                               train_data_dir=cfg['directories']['INPUT'],
                               scripts_dir=cfg['runall']['scripts_dir'],
                               submissions_dir=cfg['runall']['submissions_dir'], cfg=cfg)
    cP.preprocess_files(bDoAll=overwrite_all)

    # Create the training set for the SVD

    cP.createTrainingFiles(filesPerLanguage,
                           overwrite_all=overwrite_all)
    #raw_input('Press a key to continue')

    print('******************TrainClass********************')
    # Training SVD and creating training files
    tM = tm.TrainingClass(train_data_dir=cfg['directories']['INPUT'],
                          filesPerLanguageForLM=filesPerLanguageForLM,
                          filesPerLanguage=filesPerLanguage,
                          dictSizesTrain=dictSizesTrain,
                          cfg=cfg)
    tM.createOutputDirs()
    tM.fntTrainLMs(train_data_dir=cfg['directories']['OUTPUT'],
                   filesPerLanguageForLM=filesPerLanguageForLM,
                   overwrite_all=overwrite_all) #ayuda para dejar esto clean
    tM.fntCreateSVDs(args=args)


if __name__ == '__main__':
    main()
