import time
import os
import tomotopy as tp

import random

# absolute path of the curent folder
current_path = os.path.dirname(__file__)
#:allow import from outside the current directory
mallet_home = os.path.join(current_path, os.pardir, 'Notebook','new_mallet','mallet-2.0.8')
print(mallet_home)
mallet_path = os.path.join(current_path, os.pardir, 'Notebook','new_mallet','mallet-2.0.8','bin','mallet')

os.environ.update({'MALLET_HOME': mallet_home})

from abc import ABC, abstractmethod

from Preprocess import *
 
from gensim.models.coherencemodel import CoherenceModel
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pyLDAvis
import pyLDAvis.gensim_models
import pandas as pd
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, reset_output, export_png
import matplotlib.colors as mcolors



class TopicModel(ABC):
    """
        Class for Topic models including training, getting word-topic distribution and document-topic distribution.
        Model implemented : LDA, LDA mallet (and CTM if time permit it)
    """

    @abstractmethod
    def __init__(self, name, preprocessing_object, num_topics=20):
        """
            The constructor of Topic model

            :param name of the model
            :param preprocessing_object: object of the class Preprocessing
            :param num_topics: number of topics wanted for the model
        """
        self.name = name
        self.preprocessing = preprocessing_object
        self.num_topics = num_topics
        self.coherence_UMass = None
        self.coherence_CV = None
        self.wordTopicDistribution = None
        self.documentTopicDistribution = None
        self.model = None
        self.corpus = None

    def get_name(self):
        """
            Getter for the name of the model

            :return: name of the model
        """
        return self.name

    def get_preprocessing(self):
        """
            Getter for the Preprocessing Object

            :return: the Preprocessing Object
        """
        return self.preprocessing

    def get_num_topics(self):
        """
            Getter for the number of topic

            :return: number of topics wanted for the training
        """
        return self.num_topics

    def get_coherence_UMass(self):
        """
            Getter for the UMass coherence

            :return: the UMass coherence related to the model
        """
        return self.coherence_UMass

    def get_coherence_CV(self):
        """
            Getter for the CV coherence

            :return: the CV coherence related to the model
        """
        return self.coherence_CV

    def get_word_Topic_Distribution(self):
        """
            Getter for the word topic distribution

            :return: word topic distribution related to the model
        """
        return self.wordTopicDistribution

    def print_word_Topic_Distribution(self):
        """
            print the word topic distribution in a better form
        """

        for (nb_topic, topic) in enumerate(self.wordTopicDistribution):
            print("Topic ", nb_topic)
            print(topic)

    def get_document_Topic_Distribution(self):
        """
            Getter for the document topic distribution

            :return: document topic distribution related to the model
        """
        return self.documentTopicDistribution

    def get_model(self):
        """
            Getter for the model

            :return: Topic model
        """
        return self.model

    def get_corpus(self):
        """
            Getter for the corpus

            :return: corpus
        """
        return self.corpus

    def set_name(self, name):
        """
            Setter for the name of the model

            :param name: name of the model
        """
        self.name = name

    def set_preprocessing(self, preprocessing):
        """
            Setter for the Preprocessing Object

            :param preprocessing: Preprocessing Object
        """
        self.preprocessing = preprocessing

    def set_num_topics(self, num_topics):
        """
            Setter for the number of topic

            :param: number of topics wanted for the training
        """
        self.num_topics = num_topics

    def set_coherence_UMass(self, UMass_coherence):
        """
            Setter for the UMass coherence

            :param UMass_coherence: the UMass coherence related to a model
        """
        self.coherence_UMass = UMass_coherence

    def set_coherence_CV(self, CV_coherence):
        """
            Setter for the CV coherence

            :param CV_coherence: the CV coherence related to a model
        """
        self.coherence_CV = CV_coherence

    def set_word_Topic_Distribution(self, word_Topic_Distribution):
        """
            Setter for the word topic distribution

            :param word_Topic_Distribution: word topic distribution related to a model
        """
        self.wordTopicDistribution = word_Topic_Distribution

    def set_document_Topic_Distribution(self, document_Topic_Distribution):
        """
            Setter for the document topic distribution

            :param document_Topic_Distribution: document topic distribution related to a model
        """
        self.documentTopicDistribution = document_Topic_Distribution

    def set_model(self, model):
        """
            Setter for the model

            :param model: Topic model
        """
        self.model = model

    def set_corpus(self, corpus):
        """
            Setter for the corpus

            :param corpus: corpus
        """
        self.corpus = corpus

    def topic_visualisation(self):
        """
            Plotting the pyLDAvis figure
        """
        vis = pyLDAvis.gensim_models.prepare(gensim.models.wrappers.ldamallet.malletmodel2ldamodel(self.model),
                                             self.corpus, self.preprocessing.get_dictionary())
        pyLDAvis.display(vis)

    def topic_repartition(self):

        """
            Plot an histogram of the topic repartition for the training document. A topic is related to a document if this topic
            has the highest probability inside the document-topic distribution
        """

        topic_distribution = np.zeros(self.num_topics)
        for doc in self.documentTopicDistribution:
            topic_distribution[doc.index(max(doc))] += 1

        fig, ax = plt.subplots(figsize=(12, 6))

        # the histogram of the data

        patches = ax.bar(np.arange(len(topic_distribution)), topic_distribution)
        # Label
        ax.set_xlabel('Topic ID', fontsize=15)
        ax.set_ylabel('Topic Contribution', fontsize=15)
        ax.set_title("Topic Distribution", fontsize=20)
        ax.set_xticks(np.linspace(-1, len(topic_distribution), len(topic_distribution) + 2))

        # For each topic, we will put the percentage of the document related this topic according to the LDA model
        rects = ax.patches
        for rect, label in zip(rects, topic_distribution):
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, height + 5,
                    str(label) + "(" + str(round(label / sum(topic_distribution), 2)) + "%)", ha='center', va='bottom')
        fig.tight_layout()
        plt.show()

    def LDA_topic_visualization(self, filename):

        """
            Function inspired by : https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/
            Plot the TSNE-clustering to represent the 2D projection of each document and clustering them based on the topics where the probability is the highest inside the document-topic distribution

            :param filename: path to save the figure
        """

        # Array of topic weights
        arr = pd.DataFrame(self.documentTopicDistribution).fillna(0).values

        # Keep the well separated points (optional)
        # arr = arr[np.amax(arr, axis=1) > 0.35]

        # Dominant topic number in each doc
        topic_num = np.argmax(arr, axis=1)

        # tSNE Dimension Reduction
        tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
        tsne_lda = tsne_model.fit_transform(arr)

        # Plot the Topic Clusters using Bokeh
        reset_output()
        output_notebook()

        # Creation of a random color list
        mycolors = []
        while len(mycolors) < self.num_topics:
            color = '#%06X' % random.randint(0, 0xFFFFFF)
            if color not in mycolors:
                mycolors.append(color)
        mycolors = np.array(mycolors)

        plot = figure(title="t-SNE Clustering of {} LDA Topics".format(self.num_topics),
                      plot_width=900, plot_height=700)
        plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
        show(plot)

        reset_output()
        output_notebook()

        export_png(plot, filename=filename)

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

        self.name = temp.get_name()
        self.preprocessing = temp.get_preprocessing()
        self.num_topics = temp.get_num_topics()
        self.corpus = temp.get_corpus()
        self.documentTopicDistribution = temp.get_document_Topic_Distribution()
        self.wordTopicDistribution = temp.get_word_Topic_Distribution()
        self.model = temp.get_model()
        self.coherence_CV = temp.get_coherence_CV()
        self.coherence_UMass = temp.get_coherence_UMass()


    @abstractmethod
    def train(self, iteration, corpus="BoW"):
        """
            Training of the model

            :param iteration: number of iteration wanted to train the model
            :param corpus: "BoW" or "TfIdf" as corpus
        """
        pass

    @abstractmethod
    def compute_document_Topic_Distribution(self):
        """Computation of the document topic distribution"""
        pass

    @abstractmethod
    def compute_Word_Topic_Distribution(self):
        """Computation of the word topic distribution and redesign to be a dictionary with word as key and coefficient as value"""
        pass

class LDA(TopicModel):
    """Class for LDA"""

    def __init__(self, name, preprocessing_object, num_topics = 20):
        """
            The constructor of Topic model

            :param name : name of the model
            :param preprocessing_object: object of the class Preprocessing
            :param num_topics: number of topics wanted for the model
        """
        TopicModel.__init__(self, name, preprocessing_object, num_topics)

    def train(self, iteration, corpus = "BoW"):
        """
            Training of the LDAMallet model

            :param iteration: number of iteration wanted to train the model
            :param corpus: "BoW" or "TfIdf" as corpus
        """

        t1 = time.time()

        if corpus == "BoW":
            self.corpus = self.preprocessing.get_BoW()
        else:
            self.corpus = self.preprocessing.get_TfIdf()

        # Training model
        self.model = gensim.models.LdaMulticore(self.corpus,
                                                num_topics=self.num_topics,
                                                id2word=self.preprocessing.get_dictionary(), iterations=iteration,
                                                minimum_probability=0.0, random_state=2)

        t2 = time.time()
        print("Time to train LDA model :", t2 - t1, "sec")

        self.coherence_UMass = CoherenceModel(model=self.model, corpus=self.preprocessing.get_BoW(),
                                              coherence='u_mass').get_coherence()
        print("UMass Coherence of the model :", self.coherence_UMass)

        self.coherence_CV = CoherenceModel(model=self.model, corpus=self.preprocessing.get_BoW(),
                                           texts=self.preprocessing.get_text_preprocessed(),
                                           coherence='c_v').get_coherence()
        print("CV Coherence of the model :", self.coherence_CV)

    def compute_document_Topic_Distribution(self):
        """Computation of the document topic distribution"""
        temp = [self.model.get_document_topics(doc, minimum_probability=0.0) for doc in self.model[self.preprocessing.get_BoW()]]
        self.documentTopicDistribution = [elem[1] for elem in temp]

    def compute_Word_Topic_Distribution(self):
        """Computation of the word topic distribution and redesign to be a dictionary with word as key and coefficient as value"""
        wordTopicDistribution = []
        for _, topic in self.model.print_topics(-1):

            # We use the delimiter + to separate each term of the equation
            equation = topic.split('+')

            # Now we want to transform coeff*word to word : coeff
            temp = {}
            for eq in equation:
                # We use the delimiter * to separate the word and his coefficient
                eq = eq.split('*')

                # We will remove " character that is around each word
                temp[eq[1].replace('"', "")] = float(eq[0])
            wordTopicDistribution.append(temp)
        self.wordTopicDistribution = wordTopicDistribution


class LDAMallet(TopicModel):
    """Class for LDAMallet"""

    def __init__(self, name, preprocessing_object, num_topics = 20, prefix=" "):
        """
            The constructor of Topic model

            :param name of the model
            :param preprocessing_object: object of the class Preprocessing
            :param num_topics: number of topics wanted for the model
            :param prefix: path to saved the model without having a problem of deteted files
        """
        TopicModel.__init__(self, name, preprocessing_object, num_topics)
        self.prefix = prefix

    def train(self, iteration, corpus = "BoW"):
        """
            Training of the LDA model
            :param iteration: number of iteration wanted to train the model
            :param corpus: "BoW" or "TfIdf" as corpus
        """

        t1 = time.time()

        if corpus == "BoW":
            self.corpus = self.preprocessing.get_BoW()
        else:
            self.corpus = self.preprocessing.get_TfIdf()

        # Training model
        self.model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=self.corpus,
                                                      num_topics=self.num_topics,
                                                      id2word=self.preprocessing.get_dictionary(),
                                                      iterations=iteration)
        # Add prefix=self.prefix if it works (/temp problem)

        t2 = time.time()
        print("Time to train LDA model :", t2 - t1, "sec")

        self.coherence_UMass = CoherenceModel(model=self.model, corpus=self.preprocessing.get_BoW(),
                                              coherence='u_mass').get_coherence()
        print("UMass Coherence of the model :", self.coherence_UMass)

        self.coherence_CV = CoherenceModel(model=self.model, corpus=self.preprocessing.get_BoW(),
                                           texts=self.preprocessing.get_text_preprocessed(),
                                           coherence='c_v').get_coherence()
        print("CV Coherence of the model :", self.coherence_CV)

    def compute_document_Topic_Distribution(self):
        """Computation of the document topic distribution"""
        self.documentTopicDistribution = [[elem[1] for elem in doc]for doc in self.model[self.preprocessing.get_BoW()]]

    def compute_Word_Topic_Distribution(self):
        """Computation of the word topic distribution and redesign to be a dictionary with word as key and coefficient as value"""
        wordTopicDistribution = []
        for _, topic in self.model.print_topics(-1):

            # We use the delimiter + to separate each term of the equation
            equation = topic.split('+')

            # Now we want to transform coeff*word to word : coeff
            temp = {}
            for eq in equation:
                # We use the delimiter * to separate the word and his coefficient
                eq = eq.split('*')

                # We will remove " character that is around each word
                temp[eq[1].replace('"', "")] = float(eq[0])
            wordTopicDistribution.append(temp)
        self.wordTopicDistribution = wordTopicDistribution


class LDAtomotopy(TopicModel):
    """Class for LDA tomotopy"""

    def __init__(self, name, preprocessing_object, num_topics=20):
        """
            The constructor of Topic model

            :param name of the model
            :param preprocessing_object: object of the class Preprocessing
            :param num_topics: number of topics wanted for the model
            :param prefix: path to saved the model without having a problem of deteted files
        """
        TopicModel.__init__(self, name, preprocessing_object, num_topics)
        self.model = tp.LDAModel(k=num_topics)
        for doc in self.preprocessing.get_text_preprocessed():
            self.model.add_doc(doc)

    def train(self, iteration, corpus="BoW"):
        """
            Training of the LDA model
            :param iteration: number of iteration wanted to train the model
            :param corpus: "BoW" or "TfIdf" as corpus
        """

        t1 = time.time()

        mdl_body.train(iteration)

        t2 = time.time()
        print("Time to train LDA model :", t2 - t1, "sec")

        self.coherence_UMass = CoherenceModel(model=self.model, corpus=self.preprocessing.get_BoW(),
                                              coherence='u_mass').get_coherence()
        print("UMass Coherence of the model :", self.coherence_UMass)

        self.coherence_CV = CoherenceModel(model=self.model, corpus=self.preprocessing.get_BoW(),
                                           texts=self.preprocessing.get_text_preprocessed(),
                                           coherence='c_v').get_coherence()
        print("CV Coherence of the model :", self.coherence_CV)

    def compute_document_Topic_Distribution(self):
        """Computation of the document topic distribution"""
        self.documentTopicDistribution = list(self.model.docs.get_topic_dist(normalize=True))

    def compute_Word_Topic_Distribution(self):
        """Computation of the word topic distribution and redesign to be a dictionary with word as key and coefficient as value"""
        wordTopicDistribution = []
        for k in range(self.model.k):
            wordTopicDistribution.append({word:coeff for word,coeff in mdl_body.get_topic_words(k, top_n=10)})
        self.wordTopicDistribution = wordTopicDistribution







