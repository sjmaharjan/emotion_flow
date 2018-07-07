# Emotion Flow
---

Suraj Maharjan, Sudipta Kar, Manuel Montes-y-Gómez, Fabio A. González, and  Thamar Solorio, [Letting Emotions Flow: Success Prediction by Modeling the Flow of Emotions in Books](http://www.aclweb.org/anthology/N18-2042) (NAACL'18)


# Project Structure
---

* code
    * EmotionFlow.ipynb:  loads the model and predicts on the test data with analysis

* data
    * data_1000 :  book corpus with all book truncated to first 1K sentences
    * data_all : book corpus with all book content
    * books_meta_info.tsv : tsv file with split, genre, class label information for all books

* models
    * sf_mt_50_checked_model.hdf5 : saved best model (with whole book content, 50 chunks, MT) 

* vectors
    * nrc_mean_sd_50_all.pkl : emotion chunk feature vectors built with NRC emotion lexicons (50 chunks, whole book content)
    * nrc_mean_sd_50_all_index.pkl : book ids
    

# Cite
---

<pre>
@InProceedings{maharjan-EtAl:2018:N18-2,
  author    = {Maharjan, Suraj  and  Kar, Sudipta  and  Montes, Manuel  and  Gonzalez, Fabio A.  and  Solorio, Thamar},
  title     = {Letting Emotions Flow: Success Prediction by Modeling the Flow of Emotions in Books},
  booktitle = {Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers)},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  pages     = {259--265},
  abstract  = {Books have the power to make us feel happiness, sadness, pain, surprise, or sorrow. An author's dexterity in the use of these emotions captivates readers and makes it difficult for them to put the book down. In this paper, we model the flow of emotions over a book using recurrent neural networks and quantify its usefulness in predicting success in books. We obtained the best weighted F1-score of 69% for predicting books' success in a multitask setting (simultaneously predicting success and genre of books).},
  url       = {http://www.aclweb.org/anthology/N18-2042}
}
</pre>