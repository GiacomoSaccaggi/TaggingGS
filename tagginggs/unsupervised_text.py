# -*- coding: utf-8 -*-
"""
UNSUPERVISED TEXT
"""


class LDAText(object):
    """
    ChatBotModelEvaluate
    """
    def __init__(self, lan: 'Language in ISO 639-1', analysis_token):
        import pickle
        import os
        self.all_spacy_lan = ['zh', 'da', 'nl', 'en', 'fr', 'de', 'el', 'it', 'ja', 'lt', 'nb', 'pl', 'pt', 'ro', 'ru',
                              'af', 'sq', 'ar', 'hy', 'eu', 'bn', 'bg', 'ca', 'hr', 'cs', 'et', 'fi', 'gu', 'he', 'hi',
                              'id', 'ga', 'kn', 'ko', 'lv', 'lij', 'lb', 'mk', 'ml', 'mr', 'ne', 'fa', 'sa', 'sr', 'si',
                              'sl', 'sk', 'es', 'sv', 'hu', 'tl', 'ta', 'tt', 'te', 'th', 'tr', 'uk', 'ur', 'vi', 'yo']
        self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'

        self.nlp, self.stop_words = self._select_language(lan, 'md')
        self.analysis_token = analysis_token
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path) + '/Uploads'
        self.folder_to = f'{dir_path}/{self.analysis_token}/unsupervised'
        try:
            os.mkdir(self.folder_to)
        except FileExistsError:
            pass
        try:
            with open(self.folder_to + '/tfidf.pkl', 'rb') as f:
                self.tfidf = pickle.load(f)
            with open(self.folder_to + '/lda.pkl', 'rb') as f:
                self.lda = pickle.load(f)
        except FileNotFoundError:
            pass

    def _select_language(self, lan: 'Language in ISO 639-1', dict_size: "['sm', 'md', 'lg', 'trf']" = 'sm'):
        """
        Internal function for selecting languages
        :param lan: Language in ISO 639-1
        :param dict_size: size of dictionary
        :return: spacy dict and stopwords
        """
        import spacy
        import subprocess
        if lan in self.all_spacy_lan:
            nlp = 'not find'
            STOP_WORDS = ''
            try:
                nlp = spacy.load(f'{lan}_core_web_{dict_size}')
            except Exception as e:
                if nlp != 'not find':
                    print(e)
                pass
            if nlp == 'not find':
                try:
                    nlp = spacy.load(f'{lan}_core_news_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        print(e)
                    pass
            if nlp == 'not find':
                try:
                    subprocess.call(f'python -m spacy download {lan}_core_news_{dict_size}', shell=True)
                    nlp = spacy.load(f'{lan}_core_news_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        print(e)
                    pass
            if nlp == 'not find':
                try:
                    subprocess.call(f'python -m spacy download {lan}_core_web_{dict_size}', shell=True)
                    nlp = spacy.load(f'{lan}_core_web_{dict_size}')
                except Exception as e:
                    if nlp != 'not find':
                        print(e)
                    print('\033[91m\tError:\n\t-\tFailed download of dictionary\n\t\t-\tThe dictionary size does'
                          ' not exist\n\t\t-\tSpacy does not support this language\n\n\t\tView all languages in '
                          'website:\n\t\thttps://spacy.io/usage/models\033[0m\033[0m')
                return []
            if lan == 'zh':
                from spacy.lang.zh.stop_words import STOP_WORDS
            elif lan == 'da':
                from spacy.lang.da.stop_words import STOP_WORDS
            elif lan == 'nl':
                from spacy.lang.nl.stop_words import STOP_WORDS
            elif lan == 'en':
                from spacy.lang.en.stop_words import STOP_WORDS
            elif lan == 'fr':
                from spacy.lang.fr.stop_words import STOP_WORDS
            elif lan == 'de':
                from spacy.lang.de.stop_words import STOP_WORDS
            elif lan == 'el':
                from spacy.lang.el.stop_words import STOP_WORDS
            elif lan == 'it':
                from spacy.lang.it.stop_words import STOP_WORDS
            elif lan == 'ja':
                from spacy.lang.ja.stop_words import STOP_WORDS
            elif lan == 'lt':
                from spacy.lang.lt.stop_words import STOP_WORDS
            elif lan == 'nb':
                from spacy.lang.nb.stop_words import STOP_WORDS
            elif lan == 'pl':
                from spacy.lang.pl.stop_words import STOP_WORDS
            elif lan == 'pt':
                from spacy.lang.pt.stop_words import STOP_WORDS
            elif lan == 'ro':
                from spacy.lang.ro.stop_words import STOP_WORDS
            elif lan == 'ru':
                from spacy.lang.ru.stop_words import STOP_WORDS
            elif lan == 'es':
                from spacy.lang.es.stop_words import STOP_WORDS
            elif lan == 'af':
                from spacy.lang.af.stop_words import STOP_WORDS
            elif lan == 'sq':
                from spacy.lang.sq.stop_words import STOP_WORDS
            elif lan == 'ar':
                from spacy.lang.ar.stop_words import STOP_WORDS
            elif lan == 'hy':
                from spacy.lang.hy.stop_words import STOP_WORDS
            elif lan == 'eu':
                from spacy.lang.eu.stop_words import STOP_WORDS
            elif lan == 'bn':
                from spacy.lang.bn.stop_words import STOP_WORDS
            elif lan == 'bg':
                from spacy.lang.bg.stop_words import STOP_WORDS
            elif lan == 'ca':
                from spacy.lang.ca.stop_words import STOP_WORDS
            elif lan == 'hr':
                from spacy.lang.hr.stop_words import STOP_WORDS
            elif lan == 'cs':
                from spacy.lang.cs.stop_words import STOP_WORDS
            elif lan == 'et':
                from spacy.lang.et.stop_words import STOP_WORDS
            elif lan == 'fi':
                from spacy.lang.fi.stop_words import STOP_WORDS
            elif lan == 'gu':
                from spacy.lang.gu.stop_words import STOP_WORDS
            elif lan == 'he':
                from spacy.lang.he.stop_words import STOP_WORDS
            elif lan == 'hi':
                from spacy.lang.hi.stop_words import STOP_WORDS
            elif lan == 'hu':
                from spacy.lang.hu.stop_words import STOP_WORDS
            elif lan == 'id':
                from spacy.lang.id.stop_words import STOP_WORDS
            elif lan == 'ga':
                from spacy.lang.ga.stop_words import STOP_WORDS
            elif lan == 'kn':
                from spacy.lang.kn.stop_words import STOP_WORDS
            elif lan == 'ko':
                from spacy.lang.ko.stop_words import STOP_WORDS
            elif lan == 'lv':
                from spacy.lang.lv.stop_words import STOP_WORDS
            elif lan == 'lij':
                from spacy.lang.lij.stop_words import STOP_WORDS
            elif lan == 'lb':
                from spacy.lang.lb.stop_words import STOP_WORDS
            elif lan == 'mk':
                from spacy.lang.mk.stop_words import STOP_WORDS
            elif lan == 'ml':
                from spacy.lang.ml.stop_words import STOP_WORDS
            elif lan == 'mr':
                from spacy.lang.mr.stop_words import STOP_WORDS
            elif lan == 'ne':
                from spacy.lang.ne.stop_words import STOP_WORDS
            elif lan == 'fa':
                from spacy.lang.fa.stop_words import STOP_WORDS
            elif lan == 'sa':
                from spacy.lang.sa.stop_words import STOP_WORDS
            elif lan == 'sr':
                from spacy.lang.sr.stop_words import STOP_WORDS
            elif lan == 'si':
                from spacy.lang.si.stop_words import STOP_WORDS
            elif lan == 'sk':
                from spacy.lang.sk.stop_words import STOP_WORDS
            elif lan == 'sl':
                from spacy.lang.sl.stop_words import STOP_WORDS
            elif lan == 'sv':
                from spacy.lang.sv.stop_words import STOP_WORDS
            elif lan == 'tl':
                from spacy.lang.tl.stop_words import STOP_WORDS
            elif lan == 'ta':
                from spacy.lang.ta.stop_words import STOP_WORDS
            elif lan == 'tt':
                from spacy.lang.tt.stop_words import STOP_WORDS
            elif lan == 'te':
                from spacy.lang.te.stop_words import STOP_WORDS
            elif lan == 'th':
                from spacy.lang.th.stop_words import STOP_WORDS
            elif lan == 'tr':
                from spacy.lang.tr.stop_words import STOP_WORDS
            elif lan == 'uk':
                from spacy.lang.uk.stop_words import STOP_WORDS
            elif lan == 'ur':
                from spacy.lang.ur.stop_words import STOP_WORDS
            elif lan == 'vi':
                from spacy.lang.vi.stop_words import STOP_WORDS
            elif lan == 'yo':
                from spacy.lang.yo.stop_words import STOP_WORDS

            return nlp, STOP_WORDS
        else:
            print(f'\033[91m\tLanguage not managed, select in this languages:\n\t{",".join(self.all_spacy_lan)}\n\t'
                  'View all languages in website:\n\t\thttps://spacy.io/usage/models\033[0m')
            return []

    def __extract_words(self, sentence: str):
        """
        __extract_verbs
        """
        import re
        sentence = re.sub('[^A-Za-z]+', ' ', sentence.replace("\n", " ")).lower()
        mytokens = self.nlp(sentence)
        words = [word.lemma_.lower().strip() for word in mytokens if (word.pos_ == "VERB" or word.pos_ == "AUX"
                 or word.pos_ == "NOUN" or word.pos_ == "PROPN") and word not in self.stop_words]
        if len(words) == 0:
            return [word.lemma_.lower().strip() for word in mytokens]
        else:
            return words

    def train_lda(self, to_tag: dict, return_model=False, **kwargs):
        """
        __create_lda
        """
        import pickle
        import os
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        texts = list(to_tag.values())
        texts_mod = [' '.join(self.__extract_words(i)) for i in texts]
        tfidf = TfidfVectorizer(max_features=5000, strip_accents='unicode', lowercase=True, min_df=0.02, max_df=0.97)
        x_transf = tfidf.fit_transform(texts_mod)
        n_topics = 0
        if 'classes' in kwargs.keys():
            n_topics = kwargs['classes']
            lda = LatentDirichletAllocation(n_components=n_topics, max_iter=200, verbose=False)
            lda.fit_transform(x_transf)
        else:
            tolleranza_modello = int(len(texts_mod)*0.15)

            # Metodo per determinare il numero ottimale dei topic della LatentDirichletAllocation
            def classify_(text):
                """
                classify_
                """
                x_ = tfidf.transform([text])
                y_proba_ = partial.transform(x_)[0]
                y_ = y_proba_.argmax()
                return y_
            lda = LatentDirichletAllocation(n_components=2, max_iter=500, verbose=False)
            lda.fit_transform(x_transf)
            for n_topics in list(np.arange(2, len(texts_mod), 1)):
                partial = LatentDirichletAllocation(n_components=n_topics, max_iter=200, verbose=False)
                partial.fit_transform(x_transf)

                # Si divide il testo per 3, si calcola l'LDA su ciascuna parte e la si
                # confronta a due a due per controllare che venga sempre lo stesso topic
                prove = [True if classify_(text[:int(len(text) / 3)]) == classify_(text[int(2 * len(text) / 3):]) or
                         classify_(text[:int(len(text) / 3)]) == classify_(
                    text[int(2 * len(text) / 3):int(2 * len(text) / 3)]) or
                         classify_(text[int(2 * len(text) / 3):int(len(text) / 3)]) == classify_(
                    text[int(2 * len(text) / 3):]) else False
                         for text in texts_mod]

                # Si tiene conto di un margine di errore che può compiere il modello
                if len([flg for flg in prove if not flg]) > tolleranza_modello:
                    break
                else:
                    lda = partial

        self.tfidf = tfidf
        self.lda = lda
        try:
            os.mkdir(self.folder_to)
        except FileExistsError:
            pass
        with open(self.folder_to + '/tfidf.pkl', 'wb') as f:
            pickle.dump(tfidf, f)
        with open(self.folder_to + '/lda.pkl', 'wb') as f:
            pickle.dump(lda, f)
        os.mkdir(self.folder_to + "/cluster")
        for i in range(n_topics):
            os.mkdir(self.folder_to + f"/cluster/topic{i}")
        for t, t_, k in zip(texts, texts_mod, to_tag.keys()):
            x = self.tfidf.transform([t_])
            y_proba = self.lda.transform(x)[0]
            with open(self.folder_to +
                      f'/cluster/topic{y_proba.argmax()}/{k.replace("upload_tmp/", "").split(".")[0]}.txt', 'w') as f:
                f.write(t)
        if return_model:
            return tfidf, lda

    def classify(self, text):
        """
        classify
        """
        text_mod = ' '.join(self.__extract_words(text))
        x = self.tfidf.transform([text_mod])
        y_proba = self.lda.transform(x)[0]
        y = y_proba.argmax()
        return y
