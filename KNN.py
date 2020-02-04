from operator import itemgetter
import numpy as np


class KNNClassifier:

    def __init__(self):
        self.points = []

    def add_point(self, point):
        if not isinstance(point, DataPoint):
            raise Exception("Point must be a DataPoint object.")
        self.points.append(point)

    def add_points(self, pts):
        if isinstance(pts, (list, np.ndarray)):
            self.points += pts
        else:
            raise Exception("Must be a list of NP array")

    def make_and_add_points(self, features, classes):
        if len(features) != len(classes):
            raise Exception("Different shape of arrays features and classes")
        for i in range(len(features)):
            self.points.append(DataPoint(features[i], classes[i]))

    def get_neighbours(self, inpoint, k):
        distances = []

        for point in self.points:
            distances.append([point, inpoint.distance_to(point)])

        knn = [x for [x, _] in sorted(distances, key=itemgetter(1))[0:k]]
        return knn

    def expected_class(self, knn):
        return max(set(knn), key=knn.count)

    def classify(self, inpoint, k):
        if not isinstance(inpoint, DataPoint):
            raise Exception("Point must be a DataPoint object.")
        return self.expected_class(self.get_neighbours(inpoint, k)).class_


class DataPoint:
    def __init__(self, features, class_):
        if isinstance(features, (list, np.ndarray)):
            self.features = features
            self.n = len(features)
        else:
            raise Exception("Features wrong type, must be a numpy array or a Python List.")
        self.class_ = class_

    def distance_to(self, point):  # returns the Manhattan distance between two DataPoint objects in n dimensional space
        if not isinstance(point, DataPoint):
            raise Exception("Point must be a DataPoint object.")
        distance = 0
        if point.n == self.n:
            for i in xrange(self.n):
                distance += abs(self.features[i] - point.features[i])
        return distance


