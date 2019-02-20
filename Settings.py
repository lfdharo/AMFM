#settings .yaml


calculatescores: #ver como añadir a esto los path de donde estan los ficheros que se usan de entrada y de salida
  MIN_COUNTS: 0  # Number of times a word must occur to be included in the SVM model
  MIN_NGRAM_ORDER: 1
  MAX_NGRAM_ORDER: 3
  NUM_MAX_CORES: 4
  MIN_LOG_PROB: -10
  STARTING_VALUE_FEATURES: 25
  NFOLDS: &folds  1 # Number of NFOLDS cross-training sets we are creating for the AM experiments

  filesPerLanguageForLM: 
    en: 
      - ASPEC/ASPEC-JE/train/train
    jp: 
      - ASPEC/ASPEC-JE/train/train
  filesPerLanguage: 
    en-jp: 
     - ASPEC/ASPEC-JE/train/train-1
     - ASPEC/ASPEC-JE/train/train-2
     - ASPEC/ASPEC-JE/train/train-3
    jp-en: 
      - ASPEC/ASPEC-JE/train/train-1
      - ASPEC/ASPEC-JE/train/train-2
      - ASPEC/ASPEC-JE/train/train-3
  submissionsPerLanguagePerYear: 
    en-jp: 
    - dos015: #2015 problemas ya que es un numero
        reference: ASPEC/ASPEC-JE/devtest/devtest.jp.ref
        source: ASPEC/ASPEC-JE/devtest/devtest.en
        submissions: 
         - ASPEC/ASPEC-JE/devtest/devtest.jp.google
        test_id: devtest_en-jp
    jp-en: 
    - dos015: 
        reference: ASPEC/ASPEC-JE/devtest/devtest.en.ref
        source: ASPEC/ASPEC-JE/devtest/devtest.jp
        submissions: 
         - ASPEC/ASPEC-JE/devtest/devtest.en.google
        test_id: devtest_jp-en


general:


runall:
 NFOLDS: *folds
 dictSizesTrain: {15000: 2500}  # FOR WAT2018-My-En




preProcessModule:
trainModule:
vesctor_space:

