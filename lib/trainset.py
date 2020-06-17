import os
import time
import numpy as np
import cv2


def load_trainset(base_path="../CroppedYale", max_images_per_person=10):
    trainset = None
    labels = None
    mainfolder = list(os.walk(base_path))
    persons = mainfolder[0][1]
    # support_postfix = ["jpg", "jepg", "png", "pgm"]
    for i, personname in enumerate(persons):
        print("loading person", personname)
        personfolder = list(os.walk(base_path + r"/" + personname))[0][2]
        print(len(personfolder),max_images_per_person)

        length = min(max_images_per_person,len(personfolder))
        personfolder = personfolder[:length]

        print("there are {} images of person:{}".format(length, personname))
        person_trainset = np.zeros((length, 168 * 192))
        person_labels = []

        for j, imagetag in enumerate(personfolder):
            img = cv2.imread(base_path + r"/" + personname + r"/" + imagetag)
            img = cv2.resize(img, (168, 192), interpolation=cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            person_trainset[j] = img.reshape(1, -1)
            person_labels.append(personname)
        person_labels = np.array(person_labels, dtype=str)

        if trainset is None:
            trainset = person_trainset
            labels = person_labels
        else:
            trainset = np.vstack((trainset, person_trainset))
            labels = np.hstack((labels, person_labels))
    index = np.random.permutation(trainset.shape[0])
    trainset = trainset[index]
    labels = labels[index]

    return trainset, labels


if __name__ == '__main__':
    t1 = time.time()
    a, b = load_trainset()
    print(a.shape, b.shape)
    print(time.time() - t1)
