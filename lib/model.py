import numpy as np
from lib.trainset import load_trainset


class KNN(object):

    def __init__(self):
        self.trainX = None
        self.trainy = None

    def fit(self, mytrainX, mytrainy):
        self.trainX = mytrainX - mytrainX.mean(axis=0)
        self.trainy = mytrainy

    def predicts(self, mytestXs, dis_method="e", k=1):
        mytestXs = mytestXs - mytestXs.mean(axis=0)
        results = []
        for i in range(mytestXs.shape[0]):
            results.append(
                self.predict(
                    mytestXs[i],
                    dis_method=dis_method,
                    k=k)[0])
        return np.array(results, dtype=str)

    def predict(self, mytestx, dis_method="e", k=1):
        if self.trainX is None or self.trainy is None:
            print("your haven't fit the model")
            return
        result, sorted_distance = self.cul_distance(mytestx, dis_method, k)
        return result, sorted_distance

    def cul_distance(self, testx, dis_method="e", k=1):
        mean = testx.mean(axis=0,dtype=np.uint8)
        testx -= mean
        mat = self.trainX - testx
        if dis_method == "e" or dis_method == "E":
            mat = mat**2
            dis = mat.sum(axis=1)
        elif dis_method == "m" or dis_method == "M":
            mat = np.abs(mat)
            dis = mat.sum(axis=1)
        elif dis_method == "c" or dis_method == "C":
            m_sqrt = np.sqrt(np.sum(mat * mat, axis=1))
            tmp = np.empty_like(mat)
            for i, item in enumerate(mat):
                tmp[i] = item * m_sqrt[i]
            dis = np.sum(tmp * testx, axis=1)
        else:
            raise ValueError(
                "the distance only support e(Euclidean) and m(Manhattan) distance")
        sorted_index = dis.argsort()

        if dis_method == 'c' or dis_method == 'C':
            sorted_index = sorted_index[::-1]

        nearest_k = sorted_index[:k]
        class_count = {}
        for i in range(k):
            label = self.trainy[nearest_k[i]]
            class_count[label] = class_count.get(label, 0) + 1
        sorted_class_count = sorted(
            class_count.items(),
            reverse=True,
            key=lambda x: x[1])

        return sorted_class_count[0][0], sorted_class_count


if __name__ == '__main__':
    X, y = load_trainset("../train_dir")
    model = KNN()
    print(X.shape)

    model.fit(X[:15, ], y[:15])
    predicted_label = model.predict(X[20, ], dis_method='m')
    print(predicted_label)
    print(y[15])
