import os
import sys

current_path = os.path.dirname(__file__)

sys.path.append(os.path.join(current_path, os.pardir))

from Database.tables import *

sys.path.append(os.path.join(current_path, os.pardir, 'Class'))

from TopicModel import *
from Similarity import *
from Inference import *
from Preprocess import *


#path for the lda model for abstracts
path_abstract_model = os.path.join(current_path, os.pardir, 'Notebook', 'Saved Data', 'final_lda_abstract1_v1.pickle')
#path for the lda model for bodies
path_body_model = os.path.join(current_path, os.pardir, 'Notebook', 'Saved Data', 'final_lda_body1.pickle')

def model_run():
    current_path = os.path.dirname(__file__)

    """
    find the most similar papers to a target paper. The similar papers are selected from the database *Paper*

    :return: a list of id of papers sorted by similarity.
    """

    # path to the abstract of the target document
    f = os.listdir(os.path.join(current_path, 'static', 'uploaded_file', 'abstracts'))[0]
    abstract = os.path.join(current_path, 'static', 'uploaded_file', 'abstracts', f)

    # path to the body of the target document
    f = os.listdir(os.path.join(current_path, 'static', 'uploaded_file', 'bodies'))[0]
    body = os.path.join(current_path, 'static', 'uploaded_file', 'bodies', f)

    # open the abstract of the target document
    with open(abstract, 'r') as f:
        abstract_content = f.read()
    #open the body of the target document
    with open(body, 'r') as f:
        body_content = f.read()

    #Create Preprocessing object
    body_preprocess = Preprocessing1("body_inference", [body_content], "inference")
    body_preprocess.execute()

    abstract_preprocess = Preprocessing1("abstract_inference", [abstract_content], "inference")
    abstract_preprocess.execute()

    #Load existing model
    lda_body = LDAMallet("final_lda_body1", body_preprocess)
    lda_body.load(path_body_model)

    lda_abstract = LDAMallet("final_lda_abstract1_v1", abstract_preprocess)
    lda_abstract.load(path_abstract_model)

    # Create Inference object
    inference_body = Inference('inference_body', lda_body, body_preprocess)
    inference_abstract = Inference('inference_abstract', lda_abstract, abstract_preprocess)

    #Create Similarity object
    similarity = SimilarityV1V3('Similarity', lda_abstract, lda_body, inference_abstract,
                                inference_body)

    return similarity.most_similar_sorted(0, 'JSD', 0.8)
