from dnn.net.resnet import *


class ModelFactory():

    @staticmethod
    def create_model(model_name, params, pretrained, classifier_layer_size_list, image_size):
        return globals()[model_name](params, pretrained, classifier_layer_size_list, image_size)
