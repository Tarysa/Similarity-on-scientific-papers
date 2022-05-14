import sys
import json
import os

# /home/stsapoui/Project/LDA-similarity-Project-/Class
current_path = os.path.dirname(__file__)

# /home/stsapoui/Project/LDA-similarity-Project-
sys.path.append(os.path.join(current_path, os.pardir))

from Database.tables import *
import pandas as pd
from sklearn.model_selection import train_test_split

def extract_Paper(site):
    """
    Extracting the Paper database

    :param site: open source archive choosen (Arxiv or CiteSeerX)
    :return:  database content as dataframe splited in a training set and a validation set. Also return body and abstract for all document present in the dataframe.
    """

    # Needed to link application to the database
    app = create_app()
    app.app_context().push()

    # Query to retrieve all the content database
    paper = Paper.query.filter(Paper.SourceSite == site).order_by(Paper.Id).all()

    # Initialization of the dataframe with principal metadata
    data = pd.DataFrame(columns=['Title', 'Author', 'Date', 'Journal', 'Keyword'], index=[p.Id for p in paper])

    # Filling the dataset
    for i, p in enumerate(paper):
        data['Title'][i] = p.Title
        data['Author'][i] = p.Author
        data['Date'][i] = p.Date
        data['Journal'][i] = p.Journal
        data['Keyword'][i] = p.keyword

    # Remove missed files

    if site == "CiteSeerX":
        # Path for CiteSeer data
        file_path = os.path.join(current_path, os.pardir, 'File', 'CiteSeer')

    elif site == "Arxiv":
        # Path for CiteSeer data
        file_path = os.path.join(current_path, os.pardir, 'File', 'arXiv')

    # We will split the new dataset into training and validation set
    data_train, data_val = train_test_split(data, shuffle=True, test_size=0.1, random_state=9)

    # Sort by index
    data_train = data_train.sort_index()
    data_val = data_val.sort_index()

    # Retriveing all abstracts and bodies of documents inside training set
    abstract_train = []
    body_train = []
    for doc in data_train.index:
        if site == "CiteSeerX":
            temp_doc = doc[18:]
        elif site == "Arxiv":
            temp_doc = doc
        try:
            with open(file_path + r'/abstracts/' + temp_doc, 'r', encoding="utf8") as f1:
                abstract_train.append(f1.read())
            with open(file_path + r'/bodies/' + temp_doc, 'r', encoding="utf8") as f2:
                body_train.append(f2.read())
        except Exception as e:
            # If a document is in the traning set but abstract or body is missing, it is removed from the training set
            print(temp_doc, "not found")
            data_train.drop(data_train[data_train.index == doc].index, inplace=True)
            "File not found in listdir"

    # Retriveing all abstracts and bodies of documents inside validation set
    abstract_val = []
    body_val = []
    for doc in data_val.index:
        if site == "CiteSeerX":
            temp_doc = doc[18:]
        elif site == "Arxiv":
            temp_doc = doc
        try:
            with open(file_path + r'/abstracts/' + temp_doc, 'r', encoding="utf8") as f1:
                abstract_val.append(f1.read())
            with open(file_path + r'/bodies/' + temp_doc, 'r', encoding="utf8") as f2:
                body_val.append(f2.read())
        except Exception as e:
            # If a document is in the traning set but abstract or body is missing, it is removed from the validation set
            print(temp_doc, "not found")
            data_val.drop(data_val[data_val.index == doc].index, inplace=True)
            "File not found in listdir"

    return data_train, abstract_train, body_train, data_val, abstract_val, body_val


def extract_Evaluation():
    """
    Extracting the ValidSetTaged database

    :return: database content as dataframe splited in a training set and a validation set. Also return body and abstract for all document present in the dataframe.
    """

    # Needed to link application to the database
    app = create_app()
    app.app_context().push()

    # Query to retrieve all the content database
    valid_paper = ValidSetTaged.query.order_by(ValidSetTaged.Id).all()

    # Initialization of the dataframe with principal metadata
    data = pd.DataFrame(columns=['Title', 'Author', 'Date', 'Journal', 'Field', 'Classification'],
                        index=[vp.Id for vp in valid_paper]).sort_index()

    # Filling the dataset
    for i, vp in enumerate(valid_paper):
        data['Title'][i] = vp.Title
        data['Author'][i] = vp.Author
        data['Date'][i] = vp.Date
        data['Journal'][i] = vp.Journal
        data['Field'][i] = vp.field
        data['Classification'][i] = json.loads(vp.classification)

    # Remove missed files

    # Path for CiteSeer data
    file_path = os.path.join(current_path, os.pardir, 'File', 'ValidSet')

    # Retriveing all abstracts and bodies of documents inside the dataset
    abstract = []
    body = []
    for doc in data.index:
        try:
            with open(file_path + r'/abstracts/' + doc, 'r', encoding="utf8") as f1:
                abstract.append(f1.read())
            with open(file_path + r'/bodies/' + doc, 'r', encoding="utf8") as f2:
                body.append(f2.read())
        except Exception as e:
            # If a document is in the traning set but abstract or body is missing, it is removed from the dataset
            print(doc, "not found")
            data.drop(data[data.index == doc].index, inplace=True)
            "File not found in listdir"

    return data, abstract, body