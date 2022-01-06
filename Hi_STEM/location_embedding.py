# encoding = utf-8

from gensim.models import word2vec
import gensim
import logging
import random


def train_location2vec(model_name="location2vec"):
    id_list = []

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    trajectory_cropus = []
    with open("training_passenger", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            trajectory_cropus.append(tra)

    with open("training_fishing", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            trajectory_cropus.append(tra)

    with open("training_oil", 'r') as f:
        for line in f:
            id = line.split()[0]
            id_list.append(id)
            tra = line.split()[1:]
            trajectory_cropus.append(tra)

    # build model
    random.shuffle(trajectory_cropus)

    # build model
    embedding_model = word2vec.Word2Vec(trajectory_cropus, window=5, min_count=0, size=250, iter=100)

    embedding_model.save(model_name)


if __name__ == "__main__":
    train_location2vec()
    model = gensim.models.Word2Vec.load('location2vec')
    a = model['60133']
    print(type(a))

