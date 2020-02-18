import numpy as np
from sample import Sample
from math import sqrt
from math import pi
from math import exp
from Classifier import Classifier
from operator import itemgetter
import scipy.stats


class NaiveBayesClassifer(Classifier):

    def __init__(self, one_hot = False):
        Classifier.__init__(self)
        self.one_hot = one_hot

    def split_by_class(self):
        split = dict()
        for sample in self.points:
            if sample.class_ not in split:
                split[sample.class_] = list()
            split[sample.class_].append(sample.features)
        return split

    def mean(self, features):
        return np.mean(features)

    def standard_deviation(self, numbers):
        return np.std(numbers)

    def feature_info(self, samples):
        a = [
            (self.mean(feature), self.standard_deviation(feature), len(feature)) for feature in zip(*samples)
        ]
        return a

    def class_features(self):
        split = self.split_by_class()
        summaries = dict()
        for class_, samples in split.items():
            print  zip(*samples)
            summaries[class_] = self.feature_info(samples)
        return summaries

    def prob(self, value, mean, stdv):  # Gaussian probability distribution
        try:
            expon = exp(-((value - mean) ** 2 / (2 * stdv ** 2)))
        except ZeroDivisionError:
            expon = 0
        return (1 / (stdv * sqrt(2 * pi))) * expon

    def class_probabilities(self, feature_inf, features):
        rows = sum(feature_inf[class_][0][2] for class_ in feature_inf)
        probs = dict()
        for class_, feature_summary in feature_inf.items():
            probs[class_] = feature_inf[class_][0][2] / float(rows)
            for i in range(len(feature_summary)):
                mean, stdev, _ = feature_summary[i]
                p = self.prob(features[i], mean, stdev)
                if p != 0:
                    probs[class_] *= p
        return probs

    def classify(self, inpoint):
        if not isinstance(inpoint, Sample):
            raise Exception("Point must be a DataPoint object.")
        if not self.one_hot:
            return self.class_probabilities(self.class_features(), inpoint.features)



nb = NaiveBayesClassifer()

p = []
p.append(Sample([4, 1, 3, 0, 0, 0], 'dog'))
p.append(Sample([0.5, 1, 1, 0, 0, 0], 'dog'))
p.append(Sample([5, 1, 1, 0, 0, 0], 'dog'))
p.append(Sample([1, 15, 1, 0, 0, 0], 'dog'))
p.append(Sample([5, 3, 1, 0, 0, 0], 'dog'))
p.append(Sample([0, 0, 0, 1, 1, 51], 'cat'))
p.append(Sample([0, 0, 0, 1, 21, 1], 'cat'))
p.append(Sample([0, 0, 0, 1, 1, 1], 'cat'))
p.append(Sample([0, 0, 0, 1, 5, 55], 'cat'))
p.append(Sample([0, 0, 5, 5, 0, 0], 'fish'))
p.append(Sample([0, 0, 5, 4, 0, 0], 'fish'))
p.append(Sample([0, 0, 5, 5, 0, 0], 'fish'))
p.append(Sample([0, 0, 5, 5, 0, 0], 'fish'))

nb.add_points(p)

u = Sample([11, 13, 4, 1, 0, 55], None)
print (nb.classify(u))
