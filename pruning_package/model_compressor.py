import os
from torch.autograd import Variable
import torch.optim as optim
from pruning_package.pruner import prune
import random
import torch
import numpy as np


class PruneSimilarFilter(object):
    def __init__(self, model, test_data, train_data):
        self.model = model
        self.test_data = test_data
        self.train_data = train_data
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, optimizer=None, epoches=10):
        if optimizer is None:
            optimizer = \
                optim.SGD(self.model.classifier.parameters(),
                          lr=0.0001, momentum=0.9)

        for i in range(epoches):
            print("Epoch: ", i)
            for batch, label in self.train_data:
                self.train_batch(optimizer, batch, label)
        print("Finished fine tuning.")

    def train_batch(self, optimizer, batch, label):
        self.model.zero_grad()
        self.criterion(self.model(batch), label).backward()
        optimizer.step()

    def total_num_filters(self):
        filters = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters = filters + module.out_channels
        return filters

    def find_similar_filters(self, threshold):
        similar_filters = []
        layer_index = 0
        for name, module in self.model.features._modules.items():
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                filters_vector = self.weight_to_vectors(module.weight, module.out_channels, module.in_channels)
                similar_filter = self.find_similar_filters_for_layer(filters_vector, threshold)
                similar_filters.append((layer_index, similar_filter))
            layer_index += 1
        return similar_filters

    def weight_to_vectors(self, weight, out_channels, in_channels):
        module_avg_weight_vectors = []
        for filter_index in range(out_channels):
            w = weight.data.cpu().numpy()[filter_index]
            vector = self.get_avg_weight_for_filter(w, in_channels)
            module_avg_weight_vectors.append(vector)
        return module_avg_weight_vectors

    def get_avg_weight_for_filter(self, w, in_channels):
        result  = np.zeros((w.shape[1], w.shape[2]), dtype=np.float)
        for i in range(in_channels):
            result += w[i]
        result /= in_channels
        return result.reshape(1, -1)[0].tolist()

    def cosine_sim(self, first_v, second_v):
        dot_product = np.dot(first_v, second_v)
        norm_f = np.linalg.norm(first_v)
        norm_s = np.linalg.norm(second_v)
        return dot_product / (norm_f * norm_s)

    def find_similar_filters_for_layer(self, filters_vector, similarity_threshold):
        dict = {}
        seen = set()
        for i in range(len(filters_vector)):
            lst = filters_vector[i + 1:]
            for index, item in enumerate(lst, 0):
                sim = self.cosine_sim(filters_vector[i], item)
                if sim >= similarity_threshold:
                    val = index + i + 1
                    if val in seen:
                        continue
                    else:
                        seen.add(val)
                        if i in dict:
                            l = dict[i]
                            l.append(val)
                            dict[i] = l
                        else:
                            dict[i] = [val]
        result = []
        for key, value in dict.items():
            l = [key]
            l.extend(value)
            if len(l) > 1:
                result.append(l)
        return result


    def find_candidates(self, similar_filters):
        candidates = []
        for item in similar_filters:
            node_index = item[0]
            indexes = item[1]
            for sim_class in indexes:
                sim_class.pop(random.randrange(len(sim_class)))
                l = [(node_index, i) for i in sim_class]
                candidates.extend(l)
        return candidates

    def compressor(self, similarity_threshold, final_epochs, pruned_model_path):
        self.model.train()

        for param in self.model.features.parameters():
            param.requires_grad = True

        number_of_filters = self.total_num_filters()

        print("total number of filter in the net is: {}".format(number_of_filters))
        optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        print("current similarity threshold is: {}".format(similarity_threshold))
        similar_filters = self.find_similar_filters(similarity_threshold)
        changed = False
        if len(similar_filters) >= 1:
            prune_targets = self.find_candidates(similar_filters)
            if len(prune_targets) >= 1:
                layers_prunned = {}
                for layer_index, filter_index in prune_targets:
                    if layer_index not in layers_prunned:
                        layers_prunned[layer_index] = 0
                    layers_prunned[layer_index] = layers_prunned[layer_index] + 1

                print("Layers that will be pruned", layers_prunned)
                print("Pruning ... ")

            prune_targets = sorted(prune_targets, cmp=lambda x, y: 1 if x[0] == y[0] and x[1] < y[1]
                                   else -1 if x[0] == y[0] and x[1] >= y[1]
                                   else 0 if x[0] != y[0] else 0)

            for layer_index, filter_index in prune_targets:
                tmp_model = prune(self.model, layer_index, filter_index)
            self.model = tmp_model

            changed = True
        if changed:
            print("Finished. Going to fine tune the model a bit more")
            self.train(optimizer, epoches=final_epochs)
            dir = './models/'
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(self.model.state_dict(), './models/' + pruned_model_path)

        print("model compressor is done!")




