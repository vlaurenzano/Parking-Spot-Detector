import os
import cv2
import pickle
import UserList
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from glob import glob
from sklearn import cross_validation

class ListModelPersistent(UserList.UserList):
    '''
    List Model Persistent is a class for automatically serializing and storing lists
    '''
    def __init__(self, list=[]):
        self.data_store = os.path.dirname(os.path.realpath(__file__)) + '/bin/'+ self.__class__.__name__+'.p'
        self.data = list

    def load(self):
        '''
        Loads data if it already exists
        '''
        try:
            self.data = pickle.load(open(self.data_store, "rb"))
        except:
            pass

    def save(self):
        '''
        Saves the data
        '''
        pickle.dump(self.data, open(self.data_store, "wb"))


class ParkingSpotRoi(ListModelPersistent):
    '''
    Parking spots persistent extends UserList and simply stores it's internal list as a serialized object in the bin
    '''

class ParkingSpotClassifications(ListModelPersistent):
    '''
    Parking spot classifications stores a list of classifications for an image, whether there is or isn't a car in the spot
    '''
    def __init__(self,image_name, list=[]):
        image_name = image_name.replace("/", "_")
        self.data_store = os.path.dirname(os.path.realpath(__file__)) + '/bin/'+ self.__class__.__name__+ image_name + '.p'
        self.data = list

class PersistentClassifier():
    """
    A class to persist and load our classifiers
    This class uses relative paths to store and load data from the bin folder.
    This class wraps sklearn classifiers and persists to a pickle file stored under a hash made from it's
    constructor arguments.
    This class is lazy loads the classifier so it can take it from memory when appropriate.
    Args:
        classifier_class (ABCMeta): A meta class that works with this class. Classifiers from sklearn
    Attributes:
        hashkey (str): A key to load and persist data to
        classifier_constructor (lambda): A constructor for creating the classifier
        classifier (object): Our constructed classifier
    """
    def __init__(self, classifier_class, spot_identifier, **kwargs):
        name = classifier_class.__name__
        self.hashkey = name + str(spot_identifier) + reduce(lambda carry, key: carry + key + "-" + str(kwargs[key]), sorted(kwargs), "-")
        self.classifier_constructor = lambda : classifier_class(**kwargs)
        self.classifier = None

    def load(self):
        """ Loads our classifier from memory
        Returns:
             classifier (object): a classifier
        Raises:
            FileNotFoundException: if the file is not found
        """
        self.classifier = pickle.load(open( "bin/" + self.hashkey +".p", "rb" ))
        return self.classifier

    def fit(self,features, classes, persist=True):
        """ Fits our model and persists it
        Args:
            features(np.array): our feature set
            classes(np.array): our classes corresponding to features
        """
        self.classifier = self.classifier_constructor()
        self.classifier.fit(features,classes)
        if persist:
            pickle.dump(self.classifier, open("bin/" + self.hashkey + ".p", "wb" ))

    def predict(self, data):
        """ Fits our model and persists it
        Args:
            data(np.array): our feature set
        Returns:
            np.array: our result set
        """
        return self.load().predict(data)

    def score(self, features, classes):
        """ Returns our accuracy
        Args:
            features(np.array): our feature set
            classes(np.array): our classes corresponding to features
        """
        return self.classifier.score(features,classes)

class ParkingSpotPredictor():

    def __init__(self):
        self.parking_spots = ParkingSpotRoi()
        self.parking_spots.load()

    def _process_image(self, filename, parking_spot):
        '''
        Processes an image to be returned as a a feature
        This crops an roi for the parking spot, runs edge detection on the image and returns a sum of remaining pixels
        :param filename: opens image
        :param parking_spot: an roi given by two (x,y) positions (corners)
        :return: np.float32
        '''
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[parking_spot[0][1]:parking_spot[1][1], parking_spot[0][0]:parking_spot[1][0]]
        img = cv2.Canny(img, 127, 230)
        return np.sum(img)

    def _load_classifier(self, parking_spot):
        classifier = PersistentClassifier(RandomForestClassifier, parking_spot)
        return classifier

    def _load_parking_spot_classes(self, filename):
        '''
        Load the parking spots for this particular image
        :param filename: the filename of the image
        :return: ParkingSpotClassifications
        '''
        x = ParkingSpotClassifications(filename)
        x.load()
        return x

    def fit(self, folder, model=False):
        for i, spot in enumerate(self.parking_spots):
            image_files = glob('images' + '/*.jpg')
            classifier = self._load_classifier(i)
            features = []
            labels  = []
            for f in image_files:
                classes = self._load_parking_spot_classes(f)
                feature = self._process_image(f, spot)
                features.append([feature])
                labels.append(classes[i])

            if model:
                train_x, test_x, train_y, test_y = cross_validation.train_test_split(features, labels, test_size=.30)
                classifier.fit(train_x,train_y,False)
                print "Score for parking spot: ", i, classifier.score(test_x,test_y)
            else:
                #f,c = data[:,0:1], data[:,2]
                classifier.fit(features,labels,True)

    def predict(self, filename, parking_spot, index):
        '''
        Predict whether this parking spot is available
        :param parking_spot: Parking spot number
        :param filename: the filename
        :return:
        '''
        data = self._process_image(filename, parking_spot)
        classifier = self._load_classifier(index)
        return classifier.predict([[data]])



