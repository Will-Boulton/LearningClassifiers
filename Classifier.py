from sample import Sample


class Classifier:
    def __init__(self):
        self.points = []
        self.classes = set([])

    def add_point(self, point):
        if not isinstance(point, Sample):
            raise Exception("ERROR::POINT MUST BE AN INSTANCE OF DataPoint CLASS")
        self.points.append(point)
        self.classes.add(point.class_)

    def add_points(self, pts):
        for pt in pts:
            self.add_point(pt)

    def make_and_add_points(self, features, classes):
        if len(features) != len(classes):
            raise Exception("ERROR::MISMATCHED LENGTHS OF FEATURES AND LABELS")
        for i in range(len(features)):
            self.add_point(Sample(features[i], classes[i]))

    def classify(self, inpoint):
        pass
