from Similarity import similarity, Wu_Palmer_similarity
from Preprocess import *


def analysis(model, iteration, corpus="BoW"):
    """
        Train different topic model with a specific number of iteration and with different number of topics

        :param model: TopicModel object
        :param iteration: iteration wanted for the analysis
        :param corpus: corpus served to train the model
        :return: list of all trained models
    """

    model_list = []
    for nb_topic in [5, 10, 25, 50]:
        model.set_num_topics(nb_topic)
        model.train(iteration, corpus)
        model_list.append(model)
    return model_list


def topic_cosine_similarity(topicModel):
    """
        Average of all cosine distance of all word-topics distribution

        :param topicModel: topicModel object
        :return: average cosine distance between two random topics from a topic model
    """

    # Matrix of all coefficient from each word-topic distribution (row = one topic, column = each token from the dictionary)
    matrix = topicModel.get_model().get_topics()
    sim = 0
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[0]):
            sim += similarity(matrix[i], matrix[j], "cosine")
    return 2 * sim / (matrix.shape[0] * (matrix.shape[0] - 1))


def grid_search(topicModel, topic_list, iteration):
    """
        Calculate and return evaluation score for model trained with different number of topics

        :param topicModel: topicModel object
        :param topic_list: list of all topics wanted for the grid search
        :param iteration: number of iterations wanted for all trained model
        :return: grid: a dictionary with all following content
        coherence_CV: list of CV coherence for each model trained during the grid search
        coherence_UMass: list of UMass coherence for each model trained during the grid search
        model: list of each model trained during the grid search
        topic_similarity: each model trained during the grid search
        cosine_similarity: list of the cosine similarity between topics (see the function dist_cos) on each model trained during the grid search
    """

    # All list who will serve for the plot
    grid = {"coherence_CV": [], "coherence_UMass": [], "model": [], "topic_similarity": [], "cosine_similarity": []}

    for nb_topic in topic_list:
        print("Topic : ", nb_topic)
        # We train our model with this specific iteration and topic number
        topicModel.set_num_topics(nb_topic)
        topicModel.train(iteration)

        # Saving the coherence
        grid["coherence_CV"].append(topicModel.get_coherence_CV())
        grid["coherence_UMass"].append(topicModel.get_coherence_UMass())

        topicModel.compute_Word_Topic_Distribution()

        # Saving model
        grid["model"].append(topicModel)

        if isinstance(topicModel.get_preprocessing(), Preprocessing1):
            # For Synset, a measure has been implemented based on the function path_similarity

            word_topic_distribution = topicModel.get_word_Topic_Distribution()
            topic_similarity = []
            for i in range(len(word_topic_distribution)):
                for j in range(i + 1, len(word_topic_distribution)):
                    # Calculation of the similarity based on path_similarity between two topics inside a model
                    topic_similarity.append(Wu_Palmer_similarity(word_topic_distribution[i], word_topic_distribution[j]))
            # We keep the average of all similarity score (this score aims to give a indication of the overlapping between topics)
            grid["topic_similarity"].append(sum(topic_similarity) / len(topic_similarity))

            # Calculation of the cosine similarity based on the function dist_cos
        grid["cosine_similarity"].append(topic_cosine_similarity(topicModel))
    return grid
