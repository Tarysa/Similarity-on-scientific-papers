from TopicModel import *

import copy

class Inference:
    """Class of Infered (unknown) document with an existing topic model already trained"""

    def __init__(self, name, TopicModel, preprocessing_object):
        """
            Infered the document-topic distribution of unknown documents

            :param name: name of the infered corpus
            :param TopicModel: model pretrained
            :param preprocessing_object: preprocessing object with an unknown corpus
        """
        self.name = name

        #We store the topic model
        temp = copy.deepcopy(TopicModel)

        #We change the text_preprocessed who need to be the one related to unknown documents
        temp.get_preprocessing().set_text_preprocessed(preprocessing_object.get_text_preprocessed())

        #We change the id_list to correspond to the id of unknown documents
        temp.get_preprocessing().set_id_doc(preprocessing_object.get_id_doc())

        #Creation of the BoW with the preprocessed text from unknown document as the dictionary who has been used during the training of the model
        temp.get_preprocessing().create_BoW()
        temp.compute_document_Topic_Distribution()

        #Storing of this model
        self.model = temp

    def get_name(self):
        """
            Getter for the name of the model

            :return: name of the model
        """
        return self.name

    def get_model(self):
        """
            Getter for the pretrained model

            :return: the pretrained model
        """
        return self.model

    def set_name(self, name):
        """
            Setter for the name of the model

            :param name: name of the model
        """
        self.name = name

    def set_model(self, model):
        """
            Setter for the model

            :param model: Topic model
        """
        self.model = model

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
        self.model = temp.get_model()




