import nltk
import string
import gensim
import time
import pickle
import json

import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.wsd import lesk
from nltk import FreqDist

from gensim import corpora, models

from abc import ABC, abstractmethod


#Creation of the stopword list
stop = set(stopwords.words('english'))

#Creation of the punctuation list to remove
punct = set(string.punctuation)

#Creation the lemmatizer object
lemma = WordNetLemmatizer()


def remove_non_ascii(text):
    """
        Removing all non ascii caracter from a text

        :param text: string list
        :return: the same text without non ascii caracter
    """
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def convert_tag(tag):
    """
        Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets

        :param tag: tag from nltk.pos_tag
        :return: the corresponding wordnet.synsets tag
    """

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def get_wordnet_pos(treebank_tag):
    """
        Convert the tag given by nltk.pos_tag to the tag used by wordnetLematizer

        :param treebank_tag: tag from nltk.pos_tag
        :return: the corresponding wordnet.synsets tag
    """

    if treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('V'):
        return wn.VERB
    else:
        return ''


def histogram(liste, abscissa, ordinate, num=8):
    """
        Plot an histogram

        :param liste: list of document length
        :param num: number of point in abscissa
        :param abscissa: abscissa name
        :param ordinate: ordinate name
    """

    num_bins = 100
    fig, ax = plt.subplots(figsize=(12, 6));

    # the histogram of the data
    n, bins, patches = ax.hist(liste, num_bins, density=True)

    # Labels
    ax.set_xlabel(abscissa, fontsize=15)
    ax.set_ylabel(ordinate, fontsize=15)
    ax.grid()

    # Show some values on the abscissa
    ax.set_xticks(np.logspace(start=np.log10(min(liste) + 0.01), stop=np.log10(max(liste)), num=num, base=10.0))
    plt.xlim(min(liste), max(liste))
    ax.grid()
    plt.show()


class Preprocessing(ABC):
    """
    Preprocessing step includind preprocessing, creation of corpus and dictionary.
    """

    @abstractmethod
    def __init__(self, *arg):
        """The constructor of Preprocessing

        :param arg[0]: name of the object
        :param arg[1]: text as string list.
        :param arg[2]: name of document (index of the dataframe)
        """

        self.textPreprocessed = []
        self.dictionary = gensim.corpora.Dictionary()
        self.BoW = []
        self.TfIdf = []

        if len(arg) == 0:
            self.name = "default"
            self.text = []
            self.id_doc = []
        elif len(arg) == 1:
            self.name = arg[0]
            self.text = []
            self.id_doc = []
        elif len(arg) == 2:
            self.name = arg[0]
            self.text = arg[1]
            self.id_doc = []
        elif len(arg) == 3:
            self.name = arg[0]
            self.text = arg[1]
            self.id_doc = arg[2]
        else:
            return KeyError("Too much argument")

    def get_name(self):
        """
            Getter for the name of the model

            :return: the name of the model
        """
        return self.name

    def get_original_text(self):
        """
            Getter for the original text

            :return: the original text
        """
        return self.text

    def get_text_preprocessed(self):
        """
            Getter for the preprocessed text

            :return: the text preprocessed
        """
        return self.textPreprocessed

    def get_id_doc(self):
        """
            Getter for the list of ID document from the dataset

            :return: ID document list
        """
        return self.id_doc

    def get_dictionary(self):
        """
            Getter for the dictionary

            :return: the dictionary
        """
        return self.dictionary

    def get_BoW(self):
        """
            Getter for the BoW courpus

            :return: the BoW corpus
        """
        return self.BoW

    def get_TfIdf(self):
        """
            Getter for the tf-idf corpus

            :return: the tf-idf corpus
        """
        return self.TfIdf

    def set_name(self, name):
        """
            Setter for the original text

            :param name: original text
        """
        self.name = name

    def set_original_text(self, text):
        """
            Setter for the original text

            :param text: original text
        """
        self.text = text

    def set_text_preprocessed(self, text):
        """
            Setter for the preprocessed text

            :param text: preprocessed text
        """
        self.textPreprocessed = text

    def set_id_doc(self, IdDoc):
        """
            Setter for the ID document list

            :param IdDoc: ID document list from the dataset
        """
        self.id_doc = IdDoc

    def set_dictionary(self, dictionary):
        """
            Setter for the dictionary

            :param dictionary: dictionary related to the preprocessed text
        """
        self.dictionary = dictionary

    def set_BoW(self, BoW):
        """
            Setter for the BoW corpus

            :param BoW: BoW corpus related to the preprocessed text
        """
        self.BoW = BoW

    def set_TfIdf(self, TfIdf):
        """
            Setter for the tf-Idf corpus

            :param TfIdf: tf-Idf corpus related to the preprocessed text
        """
        self.TfIdf = TfIdf

    def save(self, path):
        """
            Save the object

            :param path: path where you want to save the object
        """
        pickle.dump(self, file=open(path + "/" + self.get_name() + ".pickle", "wb"))

    def load(self, path):
        """
            Load the object

            :param path: path of the object you want to load
        """
        temp = pickle.load(open(path, "rb"))

        self.text = temp.get_original_text()
        self.textPreprocessed = temp.get_text_preprocessed()
        self.id_doc = temp.get_id_doc()
        self.dictionary = temp.get_dictionary()
        self.BoW = temp.get_BoW()
        self.TfIdf = temp.get_TfIdf()

    def copy(self, preprocessing_object):
        """
            copy the object

            :param preprocessing_object: Preprocessing object
        """
        self.text = preprocessing_object.get_original_text()
        self.textPreprocessed = preprocessing_object.get_text_preprocessed()
        self.id_doc = preprocessing_object.get_id_doc()
        self.dictionary = preprocessing_object.get_dictionary()
        self.BoW = preprocessing_object.get_BoW()
        self.TfIdf = preprocessing_object.get_TfIdf()

    def create_dictionary(self):
        """
            Create the dictionary
        """
        self.dictionary = gensim.corpora.Dictionary(self.textPreprocessed)

    def create_BoW(self):
        """
            Create the BoW corpus
        """
        self.BoW = [self.dictionary.doc2bow(doc) for doc in self.textPreprocessed]

    def create_TfIdf(self):
        """
            Create the Tf-Idf corpus
        """
        tfidf_model = models.TfidfModel(self.BoW)
        self.TfIdf = tfidf_model[self.BoW]

    def remove_rare_word(self, threshold):

        """
            Removing all tokens who are not in the threshold st most common in the preprocessed text

            :param data: string to be cleaned
            :param threshold: threshold used to take the most frequent word
        """

        #Creation of frequence word list
        fd = FreqDist([t for doc in self.textPreprocessed for t in doc])

        #We only keep the threeshold most frequent word of the original data
        top_k_words = fd.most_common(threshold)
        list_word = [elem[0] for elem in top_k_words]

        new_text = []
        for (count, doc) in enumerate(self.textPreprocessed):
            print(count, end=',')
            new_text.append([t for t in doc if t in list_word])
        self.textPreprocessed = new_text

    def clean_poor_document(self, token_min):

        """
            Plot an histogram of remaining document length and remove all document who doesn't have the minimal amount of tokens

            :param token_min: minimal number of remaining token in a preprocessed document
        """
        # list of length for retained document
        doc_length = []

        new_text = []
        new_id_list = []
        for (idx, doc) in enumerate(self.textPreprocessed):
            temp = len(doc)
            if temp >= token_min:
                new_text.append(doc)
                doc_length.append(temp)
                new_id_list.append(self.id_doc[idx])

        self.textPreprocessed = new_text
        self.id_doc = new_id_list

        #Plotting Histogram of document length

        histogram(doc_length, "Document Length (tokens)","Normalized frequency of tokens")

    @abstractmethod
    def execute(self):
        pass

class Preprocessing1(Preprocessing):
    """
        Preprocessing 1 based on Synset and Lemmatization
    """

    def __init__(self, *arg):
        """The constructor of Preprocessing1

        :param arg[0]: name of the object
        :param arg[1]: text as string list.
        """
        Preprocessing.__init__(self, *arg)

    def execute(self):
        """
            Doing the preprocessing 1 on the text.
            Tokenizes the document and a list of bigram is made for them.
            Then finds the first synset for each word/tag combination.
            If a synset is not found for that combination it is skipped.
        """

        list_token = []

        # Time variable to have an idea of how many times does it take to run the pre processing
        t1 = time.time()

        for (nb_doc, doc) in enumerate(self.text):
            print(nb_doc, end=',')

            # Tokenization of a single document
            tokens = nltk.word_tokenize(remove_non_ascii(doc.lower()))
            pos = nltk.pos_tag(tokens)
            token_doc = []
            for elem in pos:
                # We only keep digital and stopword tokens
                if elem[0] not in stop:
                    token = ''
                    for caracter in elem[0]:
                        if caracter not in punct and not caracter.isdigit():
                            token += caracter
                    if len(token) >= 4:
                        if get_wordnet_pos(elem[1]) != '':
                            token_doc.append(lemma.lemmatize(token, get_wordnet_pos(elem[1])))
            list_token.append(token_doc)

        # Creation of the bigram list from the token list
        bigram = gensim.models.Phrases(list_token, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        list_token = [bigram_mod[doc] for doc in list_token]

        print("Tokenization done")

        # Initialization of a variable who will permit to have an idea of the percentage of token tranform into Synset
        nb_syn = 0
        for (nb_doc, doc) in enumerate(list_token):
            print(nb_doc, end=',')

            # We transform the list of token to a string
            s = " ".join([token for token in doc])
            temp_synset = []
            for (nb_word, w) in enumerate(doc):
                # Lesk from wordnet is used to infer the precise concept of the word based on the sentence where it is.
                syn = lesk(s, w)
                if syn is not None:
                    # Creation of the final list composed of the name related to each synset for the same reason as in the first pre process
                    temp_synset.append(syn.name())
                    nb_syn += 1
                else:
                    temp_synset.append(w)
            self.textPreprocessed.append(temp_synset)

        t2 = time.time()
        print("Time to train the second preprocessing :", t2 - t1, "sec")


class Preprocessing2(Preprocessing):
    """
        Preprocessing 2 based on Synset
    """

    def __init__(self, *arg):
        """The constructor of Preprocessing2

        :param arg[0]: name of the object
        :param arg[1]: text as string list.
        """
        Preprocessing.__init__(self, *arg)

    def execute(self):
        """
            Doing the preprocessing 2 on the text.
            Tokenizes the document and a list of bigram is made for them.
            Then finds the first synset for each word/tag combination.
            If a synset is not found for that combination it is skipped.
        """

        list_token = []

        # Time variable to have an idea of how many times does it take to run the pre processing
        t1 = time.time()
        nb_doc = 0

        for (nb_doc, doc) in enumerate(self.text):
            print(nb_doc, end=' ')

            # Tokenization of a single document
            tokens = nltk.word_tokenize(doc)
            token_doc = []
            for w in tokens:
                # We only keep digital and stopword tokens
                if (w not in stop and w not in punct and not w.isdigit()):
                    token_doc.append(w)
            list_token.append(token_doc)

        print("Tokenization done")
        # Creation of the bigram list from the token list
        bigram = gensim.models.Phrases(list_token, min_count=5, threshold=100)
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        list_token = [bigram_mod[doc] for doc in list_token]

        # Initialization of a variable who will permit to have an idea of the percentage of token not keeping the following process
        nb_none = 0
        for (nb_doc, doc) in enumerate(list_token):
            nb_doc += 1
            print(nb_doc, end=' ')
            # We transform the list of token to a string
            s = " ".join([token for token in doc])
            temp_synset = []
            for w in doc:
                # Lesk from wordnet is used to infer the precise concept of the word based on the sentence where it is.
                syn = lesk(s, w)
                if syn is not None:
                    # Creation of the final list composed of the name related to each synset for the same reason as in the first pre process
                    temp_synset.append(syn.name())
            self.textPreprocessed.append(temp_synset)

        t2 = time.time()
        print("Time to train the second preprocessing :", t2 - t1, "sec")

class Preprocessing3(Preprocessing):
    """
        Preprocessing 3 based on lemmatization
    """

    def __init__(self, *arg):
        """The constructor of Preprocessing3

        :param arg[0]: name of the object
        :param arg[1]: text as string list.
        """
        Preprocessing.__init__(self, *arg)

    def execute(self):
        """
            Doing the a classical lemmatization
        """
        # Time variable to have an idea of how many times does it take to run the pre processing
        t1 = time.time()

        for doc in self.text:
            doc_tokenized = word_tokenize(str(doc).lower())
            pos = nltk.pos_tag(doc_tokenized)

            # Lemmatization with the nltk POS tag keeping only the verb and the noun because a more in-depth analysis shows that with more categories, the dictionary is too big
            doc_cleaned = [lemma.lemmatize(elem[0], get_wordnet_pos(elem[1])) for elem in pos if
                            get_wordnet_pos(elem[1]) != '']

            # Stop word and Punctuations deletion
            doc_cleaned = " ".join([t for t in doc_cleaned if not t in stop])
            doc_cleaned = ''.join([t for t in doc_cleaned if not t in punct])

            # We will only keep token who has more than 3 characters
            self.textPreprocessed.append([t for t in doc_cleaned.split() if len(t) >= 4 and not t.isdigit()])

        t2 = time.time()
        print("Time to train the last preprocessing :", t2 - t1, "sec")
