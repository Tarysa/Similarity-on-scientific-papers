from TopicModel import *
from scipy.spatial import distance

import matplotlib.pyplot as plt
import seaborn as sns
import pylab

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from abc import ABC, abstractmethod


def remove_topic(liste, topic1, topic2):
    """
        From a tuple list like(topic1,topic2,similarity score) we remove all tuple who has topic1 as tuple[0] and topic2
        as tuple[1]

        :param liste: tuple list where each tuple is composed as (topic1,topic2,score_similarity)
        :param topic1: a topic from an LDA model trained
        :param topic2: a topic from an LDA model trained
        :return: a list without occurence of topic 1 and topic 2 inside tuples
    """

    liste_clear = []
    for elem in liste:
        if elem[0] != topic1 and elem[1] != topic2:
            liste_clear.append(elem)
    return liste_clear


def similarity(idx_doc1, idx_doc2, measure):
    """
        Calculate the distance score between two distribution depending on the measure in parameter

        :param idx_doc1: topic distribution of the first document
        :param idx_doc2: topic distribution of the second document
        :param measure: name of the distance measure wanted
        :return: similarity score
    """

    if "JSD" in measure:  # Jenson-Shannon Distance
        sim = distance.jensenshannon(idx_doc1, idx_doc2)
    elif "cosine" in measure:  # Cosine similarity
        sim = distance.cosine(idx_doc1, idx_doc2)
    elif "chebyshev" in measure:
        sim = distance.chebyshev(idx_doc1, idx_doc2)
    elif "L2" in measure:  # Euclidiean distance
        sim = distance.euclidean(idx_doc1, idx_doc2)
    elif "minkowski" in measure:
        sim = distance.minkowski(idx_doc1, idx_doc2, 3)
    elif "braycurtis" in measure:
        sim = distance.braycurtis(idx_doc1, idx_doc2)
    elif "canberra" in measure:
        sim = distance.canberra(idx_doc1, idx_doc2)
    elif "manhattan" in measure:
        sim = distance.cityblock(idx_doc1, idx_doc2)
    elif "correlation" in measure:
        try:
            sim = distance.correlation(idx_doc1, idx_doc2)
        except ValueError:
            print("nan value")
    else:
        raise Exception('unknown measure distance', measure)

    return sim

def Wu_Palmer_similarity(topic_word1, topic_word2):
    """
        Compute similarity between 2 topics based on Wu-Palmer similarity that gives us a score of similarity between two Synset.
        Note : We need to have a Synset as key of each dictionary. Wu-Palmer similarity attribute 1 if the word are the same and 0 if they are opposite.

        :param topics_word1: dictionary of word-topic distribution for a single topic
        :param topics_word2: dictionary of word-topic distribution for a single topic
        :return: the similarity score between two topics that will represent the average of all Wup similarity score between all word of each model
    """

    sim_coeff = 0
    nb_calcul = 0

    # Calculation of the maximum of each dictionary to normalize coefficent after
    sum_coeff1 = max([val for val in topic_word1.values()])
    sum_coeff2 = max([val for val in topic_word2.values()])

    for word1 in topic_word1.keys():
        for word2 in topic_word2.keys():
            try:
                # Wu-Palmer similarity with the product of the coefficient linked to words in the dictionary
                temp = wn.wup_similarity(wn.synset(word1), wn.synset(word2)) * topic_word1[word1] * topic_word2[
                    word2] / (sum_coeff1 * sum_coeff2)
            except ValueError:
                print("One of the word doesn't have an equivalent in synset : ", word1, word2,
                      "So we can't apply Wu-Palmer  with it")
            # Similarity could be None
            if temp is not None:
                sim_coeff += temp
                nb_calcul += 1
            # We return the average of all similarity score and we normalize it to represent similarity like measure distance does
    return 1 - sim_coeff / nb_calcul



def synset_translation(text):
    """
        The goal of this function is to create a dictionary composed of token from the corpus preprocessed  as key and his synset translation as value.

        :param text: preprocessed text
        :return: a dictionary of translation between word and
    """
    dict_translation = {}
    for doc in text:
        s = " ".join([token for token in doc])
        for w in doc:
            # Find the synset concept regarding the sentence context
            syn = lesk(s, w)
            if syn is not None:
                # Dictionary translation
                dict_translation[w] = syn.name()
    return dict_translation


def preprocess3_to_topic_word_Synset(synset_translation_dict, word_topic_distribution):
    """
        The goal of this function is to have the same topic_word output but with synset name as key

        :param synset_translation_dict: output from the function synset_translation()
        :param word_topic_distribution: word_topic_distribution from a LDA model trained with Preprocessing 3
        :return: same word_topic_distribution with only Synset name as key
    """
    # Creation of a dictionary to attribute to each word of the original text their Synset concept

    # Here we will use the output of topic_word with the original model and just replace all keys with their synset translation
    word_topic_synset = []
    for t in word_topic_distribution:
        temp = {}
        for w in t.keys():
            if w[-1] == ' ':
                if w[:-1] in synset_translation_dict.keys():
                    # We remove the last caracter of the string who is a space
                    temp[synset_translation_dict[w[:-1]]] = t[w]
            elif w in synset_translation_dict.keys():
                temp[synset_translation_dict[w]] = t[w]
        word_topic_synset.append(temp)
    return word_topic_synset


def search_min(liste):
    """
        Finding the minimum inside a list of tuple

        :param liste: tuple list where each tuple is composed as (topic1,topic2,score_similarity)
        :return: the tuple with the smallest similiarity score (0 = same)
    """

    return sorted(liste, key=lambda tup: tup[2])[0]


class Similarity:
    "Class to proceed to each similarity tools including finding the most similar document, measure evaluation and repartition of similarity score"

    @abstractmethod
    def __init__(self, name, abstract_model, body_model, abstract_inference, body_inference):
        """
            The constructor of Topic model

            :param name: name of the similarity object
            :param abstract_model: object of the class TopicModel related to abstracts
            :param body_model: object of the class TopicModel related to bodies
            :param abstract_inference: object of the class Inference related to abstracts
            :param body_inference: object of the class Inference related to bodies
        """
        self.name = name
        self.abstract_model = abstract_model
        self.body_model = body_model
        self.abstract_inference = abstract_inference
        self.body_inference = body_inference

        #List of index of document who have serve to train each model
        self.list_id_train = abstract_model.get_preprocessing().get_id_doc()
        self.list_id_infer = abstract_inference.get_model().get_preprocessing().get_id_doc()

    def get_name(self):
        return self.name

    def get_abstract_model(self):
        return self.abstract_model

    def get_body_model(self):
        return self.body_model

    def get_abstract_inference(self):
        return self.abstract_inference

    def get_body_inference(self):
        return self.body_inference

    def get_list_id_train(self):
        return self.list_id_train

    def get_list_id_infer(self):
        return self.list_id_infer

    def set_name(self, name):
        self.name = name

    def set_abstract_model(self, abstract_model):
        self.abstract_model = abstract_model

    def set_body_model(self, body_model):
        self.body_model = body_model

    def set_abstract_inference(self, abstract_inference):
        self.abstract_inference = abstract_inference

    def set_body_inference(self, body_inference):
        self.body_inference = body_inference

    def set_list_id_train(self, list_id_train):
        self.list_id_train = list_id_train

    def set_list_id_infer(self, list_id_infer):
        self.list_id_infer = list_id_infer

    @abstractmethod
    def similarity_score(self, body_distribution1, body_distribution2, abstract_distribution1, abstract_distribution2,
                         measure, learning_rate=0.5):
        """
            Similarity score between two document based on the document topic distribution

            :param body_distribution1: Document topic distribution for a single document with body model
            :param body_distribution2: Document topic distribution for a single document with body model
            :param abstract_distribution1: Document topic distribution for a single document with abstract model
            :param abstract_distribution2: Document topic distribution for a single document with abstract model
            :param learning_rate: it should be include between 0 and 1
            :param measure: similarity measure
        """
        pass

    def most_similar_sorted(self, id_doc_infer, measure, learning_rate, normalize = True):
        """
            Calculate all similarity between an unique document from the inference model and all document from the training set

            :param id_doc_infer: index of the document wanted to infer for similarity
            :param learning_rate: it should be include between 0 and 1
            :param measure: similarity measure
            :param normalize: True of normalization wanted False otherwise
            :return: ID doc list sorted ascending by similarity score
        """

        #We store the document topic distribution of the infered wanted document from abstract and body model
        body_distribution_inference = self.body_inference.get_model().get_document_Topic_Distribution()[id_doc_infer]
        abstract_distribution_inference = self.abstract_inference.get_model().get_document_Topic_Distribution()[id_doc_infer]

        #We store all document topic distribution from training set
        body_distribution = self.body_model.get_document_Topic_Distribution()
        abstract_distribution = self.abstract_model.get_document_Topic_Distribution()

        list_sim = []
        for i in range(len(body_distribution)):
            # We calculate the similarity score between all training documents and the validation document
            sim = self.similarity_score(body_distribution[i], body_distribution_inference, abstract_distribution[i],
                                        abstract_distribution_inference, measure, learning_rate)
            list_sim.append(sim)

        if normalize:
            # Normalization of the similarity score list
            min_list = min(list_sim)
            max_list = max(list_sim)

            # Retrieve the id of the training document with the temporary variable temp_list_id_infer and store the similarity score normalized
            tuple_id_sim = [(self.list_id_train[i], 1 - (elem - min_list) / (max_list - min_list)) for (i, elem) in
                            enumerate(list_sim)]
        else:
            tuple_id_sim = [(self.list_id_train[i], elem) for (i, elem) in enumerate(list_sim)]

        return sorted(tuple_id_sim, key=lambda tup: tup[1], reverse=True)

    def most_similar_sorted_eval(self, id_doc_infer, measure, learning_rate, normalize = True):
        """
            Calculate the similarity between a single document from the inferenced model and all other document from the same model (excluding himself)

            :param id_doc_infer: index of the document wanted to infer for similarity
            :param learning_rate: it should be include between 0 and 1
            :param measure: similarity measure
            :param normalize: True of normalization wanted False otherwise
            :return: ID doc list sorted ascending by similarity score
        """

        #We store the document topic distribution of the infered wanted document from abstract and body model
        body_distribution_inference = self.body_inference.get_model().get_document_Topic_Distribution()[id_doc_infer]
        abstract_distribution_inference = self.abstract_inference.get_model().get_document_Topic_Distribution()[id_doc_infer]

        #We store the list of id and remove the target document
        temp_list_id_infer = self.list_id_infer.copy()
        temp_list_id_infer.pop(id_doc_infer)

        #We store all document topic distribution from the infered model and remove the distribution related to the target document
        temp_body_distribution_inference = self.body_inference.get_model().get_document_Topic_Distribution().copy()
        temp_body_distribution_inference.pop(id_doc_infer)

        temp_abstract_distribution_inference = self.abstract_inference.get_model().get_document_Topic_Distribution().copy()
        temp_abstract_distribution_inference.pop(id_doc_infer)

        list_sim = []
        for i in range(len(temp_body_distribution_inference)):
            # We calculate the similarity score between all training documents and the validation document
            sim = self.similarity_score(temp_body_distribution_inference[i], body_distribution_inference, temp_abstract_distribution_inference[i],
                                      abstract_distribution_inference, measure, learning_rate)
            list_sim.append(sim)

        if normalize:
            # Normalization of the similarity score list
            min_list = min(list_sim)
            max_list = max(list_sim)

            #Retrieve the id of the training document with the temporary variable temp_list_id_infer and store the similarity score normalized
            tuple_id_sim = [(temp_list_id_infer[i], 1 - (elem - min_list) / (max_list - min_list)) for (i, elem) in
                            enumerate(list_sim)]
        else:
            tuple_id_sim = [(temp_list_id_infer[i], elem) for (i, elem) in enumerate(list_sim)]

        return sorted(tuple_id_sim, key=lambda tup: tup[1], reverse=True)

    def same_prediction(self, weights):

        """
            Creating a dataframe who gathered the ratio of same prediction between all similarity score for all pair of document from the model 1 and model 2 based according to each measure

            :param weights: a dictionary with similarity measure as keys and learning_rate choosen following the key as value
            :return: a dataframe who is composed of the ratio of same prediction between 2 similarity measure.
        """

        # Creation of the dataframe and filled of 0
        matrix_prediction = pd.DataFrame(index=weights.keys(), columns=weights.keys()).fillna(0)

        # Gathered all measure
        list_measure = list(weights.keys())

        for i in range(len(self.list_id_infer)):
            t1 = time.time()
            # We create a dictionary who will gathered all similarity score for the different distance measure
            pred = {}
            for measure in list_measure:
                # We create the index list ordered by similarity score between all training set document and the current target document
                ms = [id_doc for (id_doc, sim) in
                      self.most_similar_sorted(i, measure, weights[measure])]

                # We create a list with the position in list_id of each ID from ms
                pred[measure] = [ms.index(i_d) for i_d in self.list_id_train]
            for (j, measure1) in enumerate(list_measure):
                for measure2 in list_measure[j:]:
                    # Then for each combinaison of measure, we will calculate the manhattan score between this two vectors.
                    # EX : if two measures do the same prediction for the similar document, the position for the first pred element will be the same so the manattan distance will be 0
                    # This manattan is simply added to overestimate each prediction ratio and have a good insight of closed measure behaviour
                    dist_classement = similarity(pred[measure1], pred[measure2], "manhattan")
                    matrix_prediction[measure1][measure2] += dist_classement
                    matrix_prediction[measure2][measure1] += dist_classement
            print('step ' + str(i + 1) + '/' + str(len(body_distribution_inference)) + ' : ' + str(time.time() - t1) + 'sec')

        return matrix_prediction

    def similarity_distribution(self, id_doc_infer, measures):

        """
            Plot the similarity score distribution with all similarity measure between an single validation document and the entire training set

            :param id_doc_infer: index of the document wanted as target for similarity
            :param measures: similarity measure list
        """

        for measure in measures:
            ms = []
            for weight in np.around(np.linspace(0, 1, 11), 1):
                ms.append([sim_score[1] for sim_score in
                           self.most_similar_sorted(id_doc_infer, measure, weight)])

            data = pd.DataFrame(ms, index=np.around(np.linspace(0, 1, 11), 1)).transpose()

            for weight in np.around(np.linspace(0, 1, 11), decimals=1):
                sns.kdeplot(data[weight], label=weight)
            plt.legend()
            plt.title(measure)
            plt.show()

    def similarity_distribution_eval(self, id_doc_infer, measures):

        """
            Plot the similarity score distribution with all similarity measure between an single validation document and the entire training set

            :param id_doc_infer: index of the document wanted as target for similarity
            :param measures: similarity measure list
        """

        for measure in measures:
            ms = []
            for weight in np.around(np.linspace(0, 1, 11), 1):
                ms.append([sim_score[1] for sim_score in
                           self.most_similar_sorted_eval(id_doc_infer, measure, weight)])

            data = pd.DataFrame(ms, index=np.around(np.linspace(0, 1, 11), 1)).transpose()

            for weight in np.around(np.linspace(0, 1, 11), decimals=1):
                sns.kdeplot(data[weight], label=weight)
            plt.legend()
            plt.title(measure)
            plt.show()

    @abstractmethod
    def evaluation(self, dataframe_eval, measures):
        """
            Each doc of teh validation dataset will pass as target docuemnt. We are tring to predict the most similar document to the target.
            This is measured by the rate of same common field and the rate of at least one common tag between teh target document and the document
            predicted as the most similar.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: two dictionary of confusion matrix for each measure and each measure
        """
        pass

    @abstractmethod
    def evaluation2(self, dataframe_eval, measures):
        """
            Each doc of teh validation dataset will pass as target document. We are trying to predict the N most similar
            document to the target where N is the number of document with same field as target document inside the
            evaluation set.
             This is measured by the rate of same common field and the rate of at least one common tag between the
             target document and the N most similar documents.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: two dictionary of confusion matrix for each measure and each measure
        """
        pass

    @abstractmethod
    def repartition_similarity(self, dataframe_eval, measures):
        """
            For each field, 10 documents will pass as target document. We are trying to predict the N most similar
            document to the target where N is the number of document with same field as target document inside the
            evaluation set. We plot a colored  rectangle (color depend of the field) for the N most similar document found.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: figure according to the function
        """
        pass


class SimilarityV1V3(Similarity):
    """Class to implement version 1 and version 3. The difference will be if abstract and body have been
    trained with the same number of topics(V3) or not (V1)"""

    def __init__(self, name, abstract_model, body_model, abstract_inference, body_inference):
        """
            The constructor of Topic model

            :param name: name of the similarity object
            :param abstract_model: object of the class TopicModel related to abstracts
            :param body_model: object of the class TopicModel related to bodies
            :param abstract_inference: object of the class Inference related to abstracts
            :param body_inference: object of the class Inference related to bodies
        """
        Similarity.__init__(self, name,  abstract_model, body_model, abstract_inference, body_inference)

    def similarity_score(self, body_distribution1, body_distribution2, abstract_distribution1, abstract_distribution2, measure, learning_rate):
        """
            Similarity score between the document-topic distribution for abstract from the training set and from the
            validation set (inference) and similarity score between the document-topic distribution for body from the training set
            and from the validation set(inference). A learning rate has been added to see how giving more weight to the abstract
            score or the body score influence the output.

            :param body_distribution1: Document topic distribution for a single document with body model
            :param body_distribution2: Document topic distribution for a single document with body model
            :param abstract_distribution1: Document topic distribution for a single document with abstract model
            :param abstract_distribution2: Document topic distribution for a single document with abstract model
            :param learning_rate: it should be include between 0 and 1
            :param measure: similarity measure
            :return: similarity score according the first idea version.
        """
        return learning_rate * similarity(body_distribution1, body_distribution2, measure) + (1 - learning_rate) * \
                                        similarity(abstract_distribution1,abstract_distribution2, measure)

    def evaluation(self, dataframe_eval, measures):
        """
             Each doc of teh validation dataset will pass as target docuemnt. We are tring to predict the most similar document to the target.
             This is measured by the rate of same common field and the rate of at least one common tag between teh target document and the document
             predicted as the most similar.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: two dictionary of confusion matrix for each measure and each measure
        """

        Field_data = {measure: pd.DataFrame(columns=["Well classified field", "Badly classified field", "Weight"]) for
                      measure in measures}
        Tag_data = {measure: pd.DataFrame(columns=["Well classified tag", "Badly classified tag", "Weight"]) for measure
                    in measures}

        for measure in measures:
            print(measure)
            for weight in np.linspace(0, 1, 11):
                print(weight)

                #Creation of the confusion matrix presented as a dataframe
                field_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                          columns=["Well classified field", "Badly classified field"]).fillna(0)
                tag_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                        columns=["Well classified tag", "Badly classified tag"]).fillna(0)

                for i in range(len(self.list_id_infer)):
                    ms = self.most_similar_sorted_eval(i, measure, weight)[0][0]
                    field_ms = dataframe_eval.loc[ms].Field
                    field_doc = dataframe_eval.loc[self.list_id_infer[i]].Field

                    #Comparing the field of the most similar document and the targeted document
                    if field_ms == field_doc:
                        field_temp.loc[field_doc, "Well classified field"] += 1
                    else:
                        field_temp.loc[field_doc, "Badly classified field"] += 1

                    #Comparing the intersection of the tag list between the most similar document and the targeted document
                    common_tag = list(set(dataframe_eval.loc[ms].Classification).intersection(
                        dataframe_eval.loc[self.list_id_infer[i]].Classification))
                    if len(common_tag) != 0:
                        tag_temp.loc[field_doc, "Well classified tag"] += 1
                    else:
                        tag_temp.loc[field_doc, "Badly classified tag"] += 1

                #Normalization to have a ratio
                field_temp = field_temp.div(list(field_temp.sum(axis=1)), axis=0)
                field_temp["Weight"] = weight

                tag_temp = tag_temp.div(list(tag_temp.sum(axis=1)), axis=0)
                tag_temp["Weight"] = weight

                Field_data[measure] = pd.concat([Field_data[measure], field_temp])
                Tag_data[measure] = pd.concat([Tag_data[measure], tag_temp])

        return Field_data, Tag_data

    def evaluation2(self, dataframe_eval, measures):
        """
            Each doc of teh validation dataset will pass as target document. We are trying to predict the N most similar
            document to the target where N is the number of document with same field as target document inside the
            evaluation set.
            This is measured by the rate of same common field and the rate of at least one common tag between the
            target document and the N most similar documents.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: two dictionary of confusion matrix for each measure and each measure
        """

        Field_data = {measure: pd.DataFrame(columns=["Well classified field", "Badly classified field", "Weight"]) for
                      measure in measures}
        Tag_data = {measure: pd.DataFrame(columns=["Well classified tag", "Badly classified tag", "Weight"]) for measure
                    in measures}

        #Dictionary with field as key and the number of evaluation document related to this field as value
        nb_doc = {field: dataframe_eval[dataframe_eval["Field"] == field].shape[0] for field in
                  list(set(dataframe_eval["Field"]))}

        for measure in measures:
            print(measure)
            for weight in np.linspace(0, 1, 11):
                print(weight)

                #Creation of the confusion matrix presented as a dataframe
                field_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                          columns=["Well classified field", "Badly classified field"]).fillna(0)
                tag_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                        columns=["Well classified tag", "Badly classified tag"]).fillna(0)

                for i in range(len(self.list_id_infer)):
                    field_doc = dataframe_eval.loc[self.list_id_infer[i]].Field

                    #We store the N most similar document according to nb_doc
                    ms = [elem[0] for elem in self.most_similar_sorted_eval(i, measure, weight)[:nb_doc[field_doc]]]
                    for elem in ms:
                        field_ms = dataframe_eval.loc[elem].Field

                        # Comparing the field one of the N most similar document and the targeted document
                        if field_ms == field_doc:
                            field_temp.loc[field_doc, "Well classified field"] += 1
                        else:
                            field_temp.loc[field_doc, "Badly classified field"] += 1
                    field_temp.loc[field_doc] = field_temp.loc[field_doc] / nb_doc[field_doc]

                    #Comparing the intersection of the tag list between one of the N most similar document and the targeted document
                    common_tag = list(set(dataframe_eval.loc[ms[0]].Classification).intersection(
                        dataframe_eval.loc[self.list_id_infer[i]].Classification))
                    if len(common_tag) != 0:
                        tag_temp.loc[field_doc, "Well classified tag"] += 1
                    else:
                        tag_temp.loc[field_doc, "Badly classified tag"] += 1

                #Normalization to have a ratio
                field_temp = field_temp.div(list(field_temp.sum(axis=1)), axis=0)
                field_temp["Weight"] = weight

                tag_temp = tag_temp.div(list(tag_temp.sum(axis=1)), axis=0)
                tag_temp["Weight"] = weight

                Field_data[measure] = pd.concat([Field_data[measure], field_temp])
                Tag_data[measure] = pd.concat([Tag_data[measure], tag_temp])

        return Field_data, Tag_data

    def repartition_similarity(self, dataframe_eval, measures):
        """
            For each field, 10 documents will pass as target document. We are trying to predict the N most similar
            document to the target where N is the number of document with same field as target document inside the
            evaluation set. We plot a colored  rectangle (color depend of the field) for the N most similar document found.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: figure according to the function
        """

        #We store for each field the subdataframe with only this field
        data = {field: dataframe_eval[dataframe_eval["Field"] == field] for field in list(set(dataframe_eval["Field"]))}

        #Creation of the color list for each field
        color = {"computer": "blue", "physic": "grey", "biologie": "green", "math": "yellow"}

        # Insert legend for all colors
        patch = [mpatches.Patch(color=color[field], label=field) for field in color.keys()]

        #Dictionary with field as key and the id list of evaluation document related to this field as value
        id_doc_per_field = {
            field: [self.list_id_infer.index(elem) for elem in self.list_id_infer if elem in list(data[field].index)]
            for
            field in list(set(dataframe_eval["Field"]))}

        fig, ax = plt.subplots(len(measures), 11, figsize=(100, 50))

        nb_doc_plotted = 0

        for field in data.keys():
            #We only keep the first 10 id document in the list for each field
            for id_doc in id_doc_per_field[field][:10]:
                for i, measure in enumerate(measures):
                    for j, weight in enumerate(np.linspace(0, 1, 11)):
                        # create simple point
                        ax[i][j].plot(0)
                        ax[i][j].set_title(measure + " " + str(weight))
                        ms = [elem[0] for elem in
                              self.most_similar_sorted_eval(id_doc,
                                                            measure, weight)[:data[field].shape[0]]]
                        for nb, elem in enumerate(ms):
                            field_ms = dataframe_eval.loc[elem].Field

                            #Create a rectangle next to the previous with a color depending on the field of the similar document found
                            ax[i][j].add_patch(Rectangle((nb, nb_doc_plotted), 1, 1, color=color[field_ms]))
                nb_doc_plotted += 1
                print(nb_doc_plotted, end=",")

        return fig, ax


class SimilarityV2(Similarity):
    """Class to implement version 2.
    ..warning:  abstract and body must have been trained with the same number of topics"""

    def __init__(self, name, abstract_model, body_model, abstract_inference, body_inference):
        """
            The constructor of Topic model

            :param name: name of the similarity object
            :param abstract_model: object of the class TopicModel related to abstracts
            :param body_model: object of the class TopicModel related to bodies
            :param abstract_inference: object of the class Inference related to abstracts
            :param body_inference: object of the class Inference related to bodies
        """
        Similarity.__init__(self, name, abstract_model, body_model, abstract_inference, body_inference)
        if isinstance(abstract_model.get_preprocessing(), Preprocessing3):
            #Translation into Synset to reorganized topics as the reorganization is based on WU-Palmer
            self.body_translate = synset_translation(self.body_model.get_preprocessing().get_text_preprocessed())
            self.abstract_translate = synset_translation(self.abstract_model.get_preprocessing().get_text_preprocessed())
        else:
            #No need to translate into Synset with Preprocessing 1
            self.body_translate = None
            self.abstract_translate = None

        #Doing the translation part
        topic_translation = self.topic_translation(self.body_translate, self.abstract_translate)

        self.reorganization_topic(topic_translation)

    def similarity_score(self, body_distribution1, body_distribution2, abstract_distribution1, abstract_distribution2,
                         measure, learning_rate = 0.5):
        """
            Similarity score between the document-topic distribution for abstracts from the training set and bodies
            from the validation set and similarity score between the document-topic distribution for bodies from the
            training set and abstracts from the validation set.

            :param body_distribution1: Document topic distribution for a single document with body model
            :param body_distribution2: Document topic distribution for a single document with body model
            :param abstract_distribution1: Document topic distribution for a single document with abstract model
            :param abstract_distribution2: Document topic distribution for a single document with abstract model
            :param learning_rate: it should be include between 0 and 1
            :param measure: similarity measure
            :return: similarity score according the first idea version.
        """
        return learning_rate * similarity(body_distribution1, abstract_distribution2, measure) + (1 - learning_rate) * \
               similarity(abstract_distribution1, body_distribution2, measure)

    def topic_translation(self, synset_translation_body, synset_translation_abstract):

        """
            For the second version, as the training set and validation isn't related to the same model, topic order is
            likely to change between two model. This function aims to compute similarity between topics from the two
            model to find which topic from the model 2 is related for the model 1

            :param synset_translation_body: output from synset_translation with bodies
            :param synset_translation_abstract: output from synset_translation with abstracts
        Returns:
            list of tuples where the first element is the subject ID of model 1 and the second element is the subject ID linked to model 2
        """

        idx_translation = []
        sim_topic_list = []
        if isinstance(self.abstract_model.get_preprocessing(), Preprocessing1):
            topics_word_body = self.body_model.get_word_Topic_Distribution()
            topics_word_abstract = self.abstract_model.get_word_Topic_Distribution()
        else:
            #We transform all lemmatized keys by his synset equivalent

            topics_word_body = preprocess3_to_topic_word_Synset(synset_translation_body, self.body_model.get_word_Topic_Distribution())
            topics_word_abstract = preprocess3_to_topic_word_Synset(synset_translation_abstract, self.abstract_model.get_word_Topic_Distribution())
        for topic1 in range(len(topics_word_body)):
            for topic2 in range(len(topics_word_abstract)):
                # Calculation of all similarity between topics from the first model to the second
                sim_topic_list.append(
                    (topic1, topic2, Wu_Palmer_similarity(topics_word_body[topic1], topics_word_abstract[topic2])))

        while len(sim_topic_list) != 0:
            # We will search the two couple of topics where the similarity score is the lowest
            final_topic1, final_topic2, _ = search_min(sim_topic_list)

            # We add the topic found above to our translation list
            idx_translation.append((final_topic1, final_topic2))

            # We remove all element of the sim_topic_list where we can find at least one of the topics found above
            sim_topic_list = remove_topic(sim_topic_list, final_topic1, final_topic2)

        return idx_translation

    def reorganization_topic(self, idx_translation):

        """
            We will reorganized all validation document-topic distribution according the topic_translation above

            :param idx_translation: output from topic_translation()
        """

        # We create a copy of the distribution
        new_abstract_model = self.abstract_model.get_word_Topic_Distribution().copy()
        new_abstract_inference = self.abstract_inference.get_model().get_word_Topic_Distribution().copy()

        for old_id_topic, new_id_topic in idx_translation:
            # Reposition of each probability inside document-topic distribution based on id_translation
            new_abstract_model[old_id_topic] = self.abstract_model.get_word_Topic_Distribution()[new_id_topic]
            new_abstract_inference[old_id_topic] = self.abstract_inference.get_model().get_word_Topic_Distribution()[new_id_topic]

        self.abstract_model.set_word_Topic_Distribution(new_abstract_model)
        self.abstract_inference.get_model().set_word_Topic_Distribution(new_abstract_inference)

    def evaluation(self, dataframe_eval, measures):
        """
            Each doc of teh validation dataset will pass as target docuemnt. We are tring to predict the most similar document to the target.
            This is measured by the rate of same common field and the rate of at least one common tag between teh target document and the document
            predicted as the most similar.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: two dictionary of confusion matrix for each measure and each measure
        """

        Field_data = {measure: pd.DataFrame(columns=["Well classified field", "Badly classified field"]) for
                      measure in measures}
        Tag_data = {measure: pd.DataFrame(columns=["Well classified tag", "Badly classified tag"]) for measure
                    in measures}

        for measure in measures:

            # Creation of the confusion matrix presented as a dataframe
            field_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                      columns=["Well classified field", "Badly classified field"]).fillna(0)
            tag_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                    columns=["Well classified tag", "Badly classified tag"]).fillna(0)
            for i in range(len(self.list_id_infer)):
                ms = self.most_similar_sorted_eval(i, measure, 0.5)[0][0]
                field_ms = dataframe_eval.loc[ms].Field
                field_doc = dataframe_eval.loc[self.list_id_infer[i]].Field

                # Comparing the field of the most similar document and the targeted document
                if field_ms == field_doc:
                    field_temp.loc[field_doc, "Well classified field"] += 1
                else:
                    field_temp.loc[field_doc, "Badly classified field"] += 1

                # Comparing the intersection of the tag list between the most similar document and the targeted document
                common_tag = list(set(dataframe_eval.loc[ms].Classification).intersection(
                    dataframe_eval.loc[self.list_id_infer[i]].Classification))
                if len(common_tag) != 0:
                    tag_temp.loc[field_doc, "Well classified tag"] += 1
                else:
                    tag_temp.loc[field_doc, "Badly classified tag"] += 1

            # Normalization to have a ratio
            field_temp = field_temp.div(list(field_temp.sum(axis=1)), axis=0)

            tag_temp = tag_temp.div(list(tag_temp.sum(axis=1)), axis=0)

            Field_data[measure] = pd.concat([Field_data[measure], field_temp])
            Tag_data[measure] = pd.concat([Tag_data[measure], tag_temp])

        return Field_data, Tag_data

    def evaluation2(self, dataframe_eval, measures):
        """
            Each doc of teh validation dataset will pass as target document. We are trying to predict the N most similar
            document to the target where N is the number of document with same field as target document inside the
            evaluation set.
            This is measured by the rate of same common field and the rate of at least one common tag between the
            target document and the N most similar documents.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: two dictionary of confusion matrix for each measure and each measure
        """

        Field_data = {measure: pd.DataFrame(columns=["Well classified field", "Badly classified field"]) for
                      measure in measures}
        Tag_data = {measure: pd.DataFrame(columns=["Well classified tag", "Badly classified tag"]) for measure
                    in measures}

        #Dictionary with field as key and the number of evaluation document related to this field as value
        nb_doc = {field: dataframe_eval[dataframe_eval["Field"] == field].shape[0] for field in
                  list(set(dataframe_eval["Field"]))}

        for measure in measures:

            # Creation of the confusion matrix presented as a dataframe
            field_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                      columns=["Well classified field", "Badly classified field"]).fillna(0)
            tag_temp = pd.DataFrame(index=list(set(dataframe_eval.Field)),
                                    columns=["Well classified tag", "Badly classified tag"]).fillna(0)
            for i in range(len(self.list_id_infer)):
                field_doc = dataframe_eval.loc[self.list_id_infer[i]].Field

                # We store the N most similar document according to nb_doc
                ms = [elem[0] for elem in self.most_similar_sorted_eval(i, measure, 0.5)[:nb_doc[field_doc]]]
                for elem in ms:
                    field_ms = dataframe_eval.loc[elem].Field

                    # Comparing the field of the most similar document and the targeted document
                    if field_ms == field_doc:
                        field_temp.loc[field_doc, "Well classified field"] += 1
                    else:
                        field_temp.loc[field_doc, "Badly classified field"] += 1
                field_temp.loc[field_doc] = field_temp.loc[field_doc] / nb_doc[field_doc]

                # Comparing the intersection of the tag list between the most similar document and the targeted document
                common_tag = list(set(dataframe_eval.loc[ms[0]].Classification).intersection(
                    dataframe_eval.loc[self.list_id_infer[i]].Classification))
                if len(common_tag) != 0:
                    tag_temp.loc[field_doc, "Well classified tag"] += 1
                else:
                    tag_temp.loc[field_doc, "Badly classified tag"] += 1

            # Normalization to have a ratio
            field_temp = field_temp.div(list(field_temp.sum(axis=1)), axis=0)

            tag_temp = tag_temp.div(list(tag_temp.sum(axis=1)), axis=0)

            Field_data[measure] = pd.concat([Field_data[measure], field_temp])
            Tag_data[measure] = pd.concat([Tag_data[measure], tag_temp])

        return Field_data, Tag_data

    def repartition_similarity(self, dataframe_eval, measures):
        """
            For each field, 10 documents will pass as target document. We are trying to predict the N most similar
            document to the target where N is the number of document with same field as target document inside the
            evaluation set. We plot a colored  rectangle (color depend of the field) for the N most similar document found.

            :param dataframe_eval: dataframe of the evaluation set
            :param measures: similarity measure list
            :return: figure according to the function
        """

        #We store for each field the subdataframe with only this field
        data = {field: dataframe_eval[dataframe_eval["Field"] == field] for field in list(set(dataframe_eval["Field"]))}

        #Creation of the color list for each field
        color = {"computer": "blue", "physic": "grey", "biologie": "green", "math": "yellow"}

        # Insert legend for all colors
        patch = [mpatches.Patch(color=color[field], label=field) for field in color.keys()]

        #Dictionary with field as key and the id list of evaluation document related to this field as value
        id_doc_per_field = {
            field: [self.list_id_infer.index(elem) for elem in self.list_id_infer if elem in list(data[field].index)] for
            field in list(set(dataframe_eval["Field"]))}

        fig, ax = plt.subplots(len(measures), figsize=(100, 50))

        nb_doc_plotted = 0

        for field in data.keys():
            #We only keep the first 10 id document in the list for each field
            for id_doc in id_doc_per_field[field][:10]:
                for i, measure in enumerate(measures):
                    # create simple point
                    ax[i].plot(0)
                    ax[i].set_title(measure)
                    ms = [elem[0] for elem in
                          self.most_similar_sorted_eval(id_doc, measure, 0.5)[:data[field].shape[0]]]
                    for nb, elem in enumerate(ms):
                        field_ms = dataframe_eval.loc[elem].Field
                        # Create a rectangle next to the previous with a color depending on the field of the similar document found
                        ax[i].add_patch(Rectangle((nb, nb_doc_plotted), 1, 1, color=color[field_ms]))
                nb_doc_plotted += 1
                print(nb_doc_plotted, end=",")

        return fig, ax

