#!/bin/sh

#----------------#
# fastText model #
#----------------#
mkdir -p embeddings/fasttext
wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.bin.gz -O embeddings/fasttext/fasttext_frCc_cbow_d300.bin.gz
gzip -d embeddings/fasttext/fasttext_frCc_cbow_d300.bin.gz

#-----------------#
# word2vec models #
#-----------------#
# Create output directories if don't exist.
DIR_LEM=embeddings/word2vec/lemmatized
DIR_NON_LEM=embeddings/word2vec/non_lemmatized
mkdir -p $DIR_LEM && mkdir -p $DIR_NON_LEM

# Download the models trained on non-lemmatized text.
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin -O $DIR_NON_LEM/word2vec_frWac_cbow_d200.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_200_skip_cut100.bin -O $DIR_NON_LEM/word2vec_frWac_skipgram_d200.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin -O $DIR_NON_LEM/word2vec_frWac_skipgram_d500.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWiki_no_lem_no_postag_no_phrase_1000_cbow_cut100.bin -O $DIR_NON_LEM/word2vec_frWiki_cbow_d1000.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWiki_no_lem_no_postag_no_phrase_1000_skip_cut100.bin -O $DIR_NON_LEM/word2vec_frWiki_skipgram_d1000.bin

# Download the models trained on lemmatized text.
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_no_postag_no_phrase_500_cbow_cut100.bin -O $DIR_LEM/word2vec_frWac_lem_cbow_d500.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_no_postag_no_phrase_500_skip_cut100.bin -O $DIR_LEM/word2vec_frWac_lem_skipgram_d500.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWac_no_postag_no_phrase_700_skip_cut50.bin -O $DIR_LEM/word2vec_frWac_lem_skipgram_d700.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWiki_no_phrase_no_postag_500_cbow_cut10.bin -O $DIR_LEM/word2vec_frWiki_lem_cbow_d500.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWiki_no_phrase_no_postag_700_cbow_cut100.bin -O $DIR_LEM/word2vec_frWiki_lem_cbow_d700.bin
wget -c https://s3.us-east-2.amazonaws.com/embeddings.net/embeddings/frWiki_no_phrase_no_postag_1000_skip_cut100.bin -O $DIR_LEM/word2vec_frWiki_lem_skipgram_d1000.bin
