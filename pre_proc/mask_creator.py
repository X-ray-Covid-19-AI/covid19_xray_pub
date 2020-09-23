from keras.models import load_model

MODEL_NAME = '/nih-yonina/workspace/code/covid19_xray/pre_proc/trained_model.hdf5'

class MaskCreator(object):

    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.trained_model = load_model(MODEL_NAME)
        print("New mask created")
        return cls.__instance
    def get_model(self):
        return self.__instance.trained_model

