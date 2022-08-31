import os
import pickle
from model.model import Model

class ModelPersister:
    _BASE_PATH = "data\\models\\"

    def persist(model: Model, name):
        if len(name) > 0:
            path = os.path.join(
                ModelPersister._BASE_PATH, name)
            with open(path, 'wb', ) as file:
                pickle.dump(model, file)
        else:
            raise Exception("Please provide a name")

    def load(name):
        if len(name) > 0:
            path = os.path.join(
                ModelPersister._BASE_PATH, name)
            with open(path, 'rb', ) as file:
                return pickle.load(file)
        else:
            raise Exception("Please provide a name")
