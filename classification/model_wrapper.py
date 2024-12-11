import numpy as np
import torch
from classification.train_base import MultiPartitioningClassifier
import reverse_geocode

from classification.utils_global import accuracy


class taskPredictor :

    def __init__(self, model):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # print(self.device, type(self.device))
        print(str(self.device) == 'cuda')
        if (str(self.device) == 'cuda'):
            # print("loading model in cuda")
            self.model = model.cuda()
        else:
            # print("model is loaded in cpu")
            self.model = model

    def predict(self, x):

        if (str(self.device) == 'cuda'):
            # print("data is loaded in cuda")
            x = torch.from_numpy(x).cuda()
        else:
            # print("data is loaded in cpu")
            x = torch.from_numpy(x)

        pred_classes, pred_latitudes, pred_longitudes = self.model.inference(x)
        lat_lng_tuple = []
        # for p_key in pred_classes.keys():
        p_key = 'hierarchy'
        for pred_class, pred_lat, pred_lng in zip(
                pred_classes[p_key].cpu().numpy(),
                pred_latitudes[p_key].cpu().numpy(),
                pred_longitudes[p_key].cpu().numpy(),
        ):
            lat_lng_tuple.append((pred_lat, pred_lng))

        # print(val_tensor.shape)
        # print(len(lat_lng_tuple))

        pred_country = [i['country'].replace(' ', '').lower() for i in reverse_geocode.search(lat_lng_tuple)]

        return pred_country


    def evaluate(self,x_val, y_val, verbose=0):

        pred_country = self.predict(x_val)

        accuracy_list = []

        for i in zip(pred_country,y_val):

            accuracy_list.append((i[0] == i[1])*1)

        return np.array(accuracy_list).sum() / len(accuracy_list)





