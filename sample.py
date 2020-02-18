import numpy as np

class Sample:
    id_ = 0
    def __init__(self, features, class_):
        self.id = Sample.id_
        Sample.id_ += 1

        if isinstance(features, (list, np.ndarray)):
            self.features = features
            self.n = len(features)
        else:
            raise Exception("Features wrong type, must be a numpy array or a Python List.")
        self.class_ = class_
        self.predicted_class = "Undefined"

    def distance_to(self, point):  # returns the Manhattan distance between two DataPoint objects in n dimensional space
        if not isinstance(point, Sample):
            raise Exception("Point must be a DataPoint object.")
        distance = 0
        if point.n == self.n:
            for i in xrange(self.n):
                distance += abs(self.features[i] - point.features[i])
        return distance

    def __repr__(self):
        return '<sample_id' + str(self.id) + ' class:' + str(self.class_) +'>'