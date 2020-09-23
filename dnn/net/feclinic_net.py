import torch
import torch.nn as nn
from dnn.net.model_factory import ModelFactory
import torch.nn.functional as F


class FeClinicNet(nn.Module):

    def __init__(self, feature_extractor_name, feature_extractor_params, classifier_fc_size, pretrained, image_size):
        super(FeClinicNet, self).__init__()
        self.feature_extractor = ModelFactory.create_model(feature_extractor_name, feature_extractor_params, pretrained,
                                                           image_size)

        self.CLINIC_DATA_ITEMS = 2
        self.fc1 = nn.Linear(feature_extractor_params['fc_size'] + self.CLINIC_DATA_ITEMS, classifier_fc_size[0])
        self.fc2 = nn.Linear(classifier_fc_size[0], classifier_fc_size[1])

    def forward(self, image, tabolar_data):
        img, clinic_data = image, tabolar_data
        x1 = self.feature_extractor(img)

        x = torch.cat((x1, clinic_data), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
