# Emotion Flow
---

Suraj Maharjan, Sudipta Kar, Manuel Montes-y-Gómez, Fabio A. González, and  Thamar Solorio, [Letting Emotions Flow: Success Prediction by Modeling the Flow of Emotions in Books](http://www.aclweb.org/anthology/N18-2042) (NAACL'18)


# Project Structure
 ---

* code
    * EmotionFlow.ipynb:  loads the model and predict on the test data with analysis

* data
    * data_1000 :  book corpus with all book truncated to first 1K sentences
    * data_all : book corpus with all book content
    * books_meta_info.tsv : tsv file with split, genre, class label information for all books

* models
    * sf_mt_50_checked_model.hdf5 : saved best model (with whole book content, 50 chunks, MT) 

* vectors
    * nrc_mean_sd_50_all.pkl : emotion chunk feature vectors built with NRC emotion lexicon (50 chunks, whole book content)
    * nrc_mean_sd_50_all_index.pkl : book ids
    