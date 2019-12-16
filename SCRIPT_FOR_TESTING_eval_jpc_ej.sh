################

#In order to reproduce this results and everything works you will need to do use perlbrew to use perl 5.24.0 version:
# sudo apt get install perlbrew
# perlbrew install perl-5.24.0
# perlbrew switch perl-5.24.0
# perl -v

## download scripts

## Moses 2.1.1 (BLEU, tokenizer)
# git clone -b RELEASE-2.1.1 https://github.com/moses-smt/mosesdecoder

## RIBES 1.02.4
# It can be manually downloaded from <http://www.kecl.ntt.co.jp/icl/lirg/ribes/index.html>

## WAT scripts
# wget http://lotus.kuee.kyoto-u.ac.jp/WAT/evaluation/automatic_evaluation_systems/script.segmentation.distribution.tar.gz

#Paths for Scripts
SCRIPT=/home/enrique/Escritorio/Scripts/script.segmentation.distribution
MOSES_SCRIPT=/home/enrique/Escritorio/BLEU/mosesdecoder-master
RIBES=/home/enrique/Escritorio
INDIC_SCRIPT=/home/enrique/Escritorio/indic_nlp_library-INDIC_NLP_0.3/src/indicnlp
KYTEA_MODEL=/home/enrique/Escritorio/kytea-0.4.6/models
AMFM=/home/enrique/Escritorio/TFG_Pendrive/AMFM/AMFM_TFG2018/Test #ruta del fichero donde esta el metodo test para amfm

#Path for AMFM files
REF_AMFM=/home/enrique/Escritorio/Data_for_TFG_WAT2017_ordered/processed/ASPEC/JE  #ruta donde esta el fichero listo para amfm

#Paths to store results of testing
RESULTADOS_BLEU=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_BLEU/resultados_jpc_ej_BLEU.txt
RESULTADOS_RIBES=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_RIBES/resultados_jpc_ej_RIBES.txt
RESULTADOS_AMFM=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_AMFM/resultados_jpc_ej_AMFM.txt

RESAMFM=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_AMFM/JPO/EJ

RESULTADOS_BLEU200=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_BLEU/resultados_jpc_ej_BLEU_200.txt
RESULTADOS_RIBES200=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_RIBES/resultados_jpc_ej_RIBES_200.txt
RESULTADOS_AMFM200=/home/enrique/Escritorio/reproduce_results_20190212/RESULTADOS_AMFM/resultados_jpc_ej_AMFM_200_1000.txt

###############SUBMISSIONS###################
#2k sentences pairwise analisis
FICHERO_1=jpc-ej-1407.txt
FICHERO_2=jpc-ej-1454.txt
FICHERO_3=jpc-ej-1470.txt
FICHERO_4=jpc-ej-1406.txt
FICHERO_5=jpc-ej-1339.txt
#subs200 adecuacy analisis
FICHERO_6=sub.jpo.enja.1407.txt
FICHERO_7=sub.jpo.enja.1454.txt
FICHERO_8=sub.jpo.enja.1470.txt

##################REFERENCES FOT TESTING#######################

REF=tests/JPC-test_ja-en.ja
REF200=/home/enrique/Escritorio/reproduce_results_20190212/preProcessed_200/processed/JPC/EJ/ref.jpo.enja.ehr.txt

###############################################################

lang=jp #Change accordingly to the langueage we are testing en/jp/zh/ta/hi/ko

# only change the DATASET_DIR for choosing the models for amfm test
CFG_AMFM=/home/enrique/Escritorio/TFG_Pendrive/AMFM/AMFM_TFG2018/Test



##############################################################################################################################################
## tokenization and pre-processing

declare -a lista=( "$FICHERO_6" "$FICHERO_7" "$FICHERO_8" ) #( "$FICHERO_1" "$FICHERO_2" "$FICHERO_3" "$FICHERO_4" "$FICHERO_5" "$FICHERO_6"
for i in ${lista[@]}; #{1..3}
do

Nom=$i
echo
echo
echo We are now evaluating this: $i

if [[ "$i" == "jpc-ej-1407.txt" || "$i" == "jpc-ej-1454.txt" || "$i" == "jpc-ej-1470.txt" || "$i" == "jpc-ej-1406.txt" || "$i" == "jpc-ej-1339.txt" ]]
then 
echo " JPC EJ"
lang=jp
CFG_AMFM=/home/enrique/Escritorio/TFG_Pendrive/AMFM/AMFM_TFG2018/Test

result_org=results_org/$i
result=results/$i
tail -2000 $result_org > $result

test=tests/JPC-test_ja-en.ja #REF_2
test_tok=tests_tok/JPC-test_ja-en.ja

 cat $test | \
   perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' | \
   sh ${SCRIPT}/remove-space.sh | \
   perl ${SCRIPT}/h2z-utf8-without-space.pl | \
   kytea -model ${KYTEA_MODEL}/jp-0.4.2-utf8-1.mod -out tok | \
   perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
   perl -Mencoding=utf8 -pe 'while(s/([０-９]) ([０-９])/$1$2/g){} s/([０-９]) (．) ([０-９])/$1$2$3/g; while(s/([Ａ-Ｚ]) ([Ａ-Ｚａ-ｚ])/$1$2/g){} while(s/([ａ-ｚ]) ([ａ-ｚ])/$1$2/g){}' \
   > $test_tok

result_tok=results_tok/$i

 cat $result | \
   perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' | \
   sh ${SCRIPT}/remove-space.sh | \
   perl ${SCRIPT}/h2z-utf8-without-space.pl | \
   kytea -model ${KYTEA_MODEL}/jp-0.4.2-utf8-1.mod -out tok | \
   perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
   perl -Mencoding=utf8 -pe 'while(s/([０-９]) ([０-９])/$1$2/g){} s/([０-９]) (．) ([０-９])/$1$2$3/g; while(s/([Ａ-Ｚ]) ([Ａ-Ｚａ-ｚ])/$1$2/g){} while(s/([ａ-ｚ]) ([ａ-ｚ])/$1$2/g){}' \
   > $result_tok

echo "CALCULANDO SCORES"
echo
echo "$i" >> $RESULTADOS_BLEU
perl ${MOSES_SCRIPT}/scripts/generic/multi-bleu.perl $test_tok < $result_tok >> $RESULTADOS_BLEU
python3 ${RIBES}/RIBES-1.02.4/RIBES.py -c -r $test_tok $result_tok >> $RESULTADOS_RIBES

##long process unccoment if want to try AMFM
#
#echo "AMFM ORIG "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig.txt
echo "AMFM NEXT "
python3 ${AMFM}/Test.py $test_tok $result_tok $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokExt_l05_SVD1000.yaml > temp.txt
echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokExt_l05_SVD1000.txt
tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokExt_l05_SVD1000.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l02_SVD1000.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l02_SVD1000.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l02_SVD1000.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l04_SVD1000.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l04_SVD1000.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l04_SVD1000.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l06_SVD1000.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l06_SVD1000.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l06_SVD1000.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l08_SVD1000.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l08_SVD1000.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l08_SVD1000.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l10_SVD1000.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l10_SVD1000.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l10_SVD1000.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD0.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD0.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD0.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD200.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD200.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD200.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD400.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD400.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD400.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD600.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD600.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD600.txt
#echo "AMFM NEXT "
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD1000.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD1000.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD1000.txt
#
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD1300.yaml > temp.txt
#echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD1300.txt
#tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokOrig_l05_SVD1300.txt
#


###################################ADECUACY ANALISIS####################################################
elif [[ "$i" == "sub.jpo.enja.1407.txt" || "$i" == "sub.jpo.enja.1454.txt" || "$i" == "sub.jpo.enja.1470.txt" ]]
then 
echo " JPC EJ"
lang=jp
#CFG_AMFM=/home/enrique/Escritorio/TFG_Pendrive/AMFM/AMFM_TFG2018/Test/SettingsTest_en_jp_JPO.yaml #_4

CFG_AMFM=/home/enrique/Escritorio/TFG_Pendrive/AMFM/AMFM_TFG2018/Test

result=/home/enrique/Escritorio/reproduce_results_20190212/preProcessed_200/processed/JPC/EJ/$i

test=$REF200
test_tok=tests_tok/JPC-test_ja-en_200.ja

 cat $test | \
   perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' | \
   sh ${SCRIPT}/remove-space.sh | \
   perl ${SCRIPT}/h2z-utf8-without-space.pl | \
   kytea -model ${KYTEA_MODEL}/jp-0.4.2-utf8-1.mod -out tok | \
   perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
   perl -Mencoding=utf8 -pe 'while(s/([０-９]) ([０-９])/$1$2/g){} s/([０-９]) (．) ([０-９])/$1$2$3/g; while(s/([Ａ-Ｚ]) ([Ａ-Ｚａ-ｚ])/$1$2/g){} while(s/([ａ-ｚ]) ([ａ-ｚ])/$1$2/g){}' \
   > $test_tok

result_tok=results_tok/$i

 cat $result | \
   perl -Mencoding=utf8 -pe 's/(.)［[０-９．]+］$/${1}/;' | \
   sh ${SCRIPT}/remove-space.sh | \
   perl ${SCRIPT}/h2z-utf8-without-space.pl | \
   kytea -model ${KYTEA_MODEL}/jp-0.4.2-utf8-1.mod -out tok | \
   perl -pe 's/^ +//; s/ +$//; s/ +/ /g;' | \
   perl -Mencoding=utf8 -pe 'while(s/([０-９]) ([０-９])/$1$2/g){} s/([０-９]) (．) ([０-９])/$1$2$3/g; while(s/([Ａ-Ｚ]) ([Ａ-Ｚａ-ｚ])/$1$2/g){} while(s/([ａ-ｚ]) ([ａ-ｚ])/$1$2/g){}' \
   > $result_tok

echo "CALCULANDO SCORES"
echo
##echo "$i" >> $RESULTADOS_BLEU200
##perl ${MOSES_SCRIPT}/scripts/generic/multi-bleu.perl $test_tok < $result_tok >> $RESULTADOS_BLEU200
##python3 ${RIBES}/RIBES-1.02.4/RIBES.py -c -r $test_tok $result_tok >> $RESULTADOS_RIBES200

#long process unccoment if want to try AMFM
#python3 ${AMFM}/Test.py $test $result $lang -c ${CFG_AMFM} > temp.txt
#echo "$i" >> $RESULTADOS_AMFM200
#tail -5 temp.txt >> $RESULTADOS_AMFM200

echo "AMFM NEXT "
python3 ${AMFM}/Test.py $test_tok $result_tok $lang -c ${CFG_AMFM}/SettingsTest_jp_en_JPO_TokExt_l05_SVD1000.yaml > temp.txt
echo "$i" >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokExt_l05_SVD1000.txt
tail -5 temp.txt >> ${RESAMFM}/SettingsTest_jp_en_JPO_TokExt_l05_SVD1000.txt



fi

echo

done

#grep GLOBAL AVERAGE /home/enrique/Escritorio/resultadosAMFM.txt > algo.txt #nos guarda x3 globalam-fm-amfm
#grep AM_FM /home/enrique/Escritorio/algo.txt > Simple_results_AMFM.txt #nos quedamos con 1 solo ya


################################################################################################################################################
## calculate metrics

