from operator import itemgetter
import numpy as np
from Classifier import Classifier

from sample import Sample


class KNNClassifier(Classifier):

    def __init__(self):
        Classifier.__init__(self)

    def get_neighbours(self, inpoint, k):
        distances = []

        for point in self.points:
            distances.append([point, inpoint.distance_to(point)])

        knn = [x for [x, _] in sorted(distances, key=itemgetter(1))[0:k]]
        return knn

    def expected_class(self, knn):
        return max(set(knn), key=knn.count)

    def classify(self, inpoint, k):
        if not isinstance(inpoint, Sample):
            raise Exception("Point must be a DataPoint object.")
        return self.expected_class(self.get_neighbours(inpoint, k)).class_
