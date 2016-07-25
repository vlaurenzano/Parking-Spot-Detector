import uuid
import cv2
import models
from glob import glob

class ParkingSpotSelector:
    '''
    The Parking Spot Selector provides a simply gui for selecting parking spaces
    While running, use the mouse to drag over parking spots
    Hit the 'r' key to refresh the spots
    Hit the 'c' key to close
    Parking spots are automatically persisted

    Usage: ParkingSpotSelector('images/G0031388.JPG').start_loop()
    '''
    def __init__(self, filename):
        '''
        :param filename: string of an image path relative to the working directory
        '''
        self.image = cv2.imread(filename)
        self.clone = self.image.copy()
        self.parking_spots = models.ParkingSpotRoi()
        self.parking_spots.load()
        self.roi = None
        self.window_name = uuid.uuid4().hex


    def _select_roi(self, event, x, y, flags, param):
        '''
        Select Roi is used as a callback for selecting regions of interest (parking spots)
        :param event: a cv2 event
        :param x: the x coordinate of the mouse
        :param y: the y coordinate of the mouse
        :param flags: cv2 flags
        :param param: cv2 param
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            self.roi.append((x, y))
            self.parking_spots.append(self.roi)
            cv2.rectangle(self.image, self.roi[0], self.roi[1], (0, 255, 0), 4)
            cv2.imshow(self.window_name, self.image)

    def _reset_parking_spots(self):
        '''
        Reset our parking spots
        '''
        self.parking_spots.data = []
        self.image = self.clone.copy()


    def _setup_windows(self):
        '''
        Runs our initial set up, open the window show, show the image and draw the parking spots
        '''
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.image)
        cv2.setMouseCallback(self.window_name, self._select_roi)
        for spot in self.parking_spots:
            cv2.rectangle(self.image, spot[0], spot[1], (0, 255, 0), 4)

    def start_loop(self):
        '''
        Loop until user hits "c"
        Reset parking spots on "r"
        '''
        self._setup_windows()
        while True:
            cv2.imshow(self.window_name, self.image)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("r") or key == ord("R"):
                self._reset_parking_spots()
            elif key == ord("c") or key == ord("C"):
                break
        self.parking_spots.save()
        cv2.destroyAllWindows()




class ParkingSpotClassifier:
    '''
    The Parking Spot Selector provides a simple gui for selecting parking spaces
    While running, use the mouse to drag over parking spots
    Hit the 'r' key to refresh the spots
    Hit the 'c' key to close
    Parking spots are automatically persisted

    Usage: ParkingSpotClassifier('images/G0031388.JPG').start_loop()
    '''
    def __init__(self, filename):
        '''
        :param filename: string of an image path relative to the working directory
        '''
        self.image = cv2.imread(filename)
        self.filename = filename
        self.clone = self.image.copy()
        self.parking_spots = models.ParkingSpotRoi()
        self.parking_spots.load()
        self.parking_spot_classification = models.ParkingSpotClassifications(filename)
        self.parking_spot_classification.load()
        self.window_name = uuid.uuid4().hex


    def _toggle_spot(self, event, x, y, flags, param):
        '''
        Select Roi is used as a callback for selecting regions of interest (parking spots)
        :param event: a cv2 event
        :param x: the x coordinate of the mouse
        :param y: the y coordinate of the mouse
        :param flags: cv2 flags
        :param param: cv2 param
        '''
        refresh = False
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, roi in enumerate(self.parking_spots):
                x_bounds = sorted([roi[0][0], roi[1][0]])
                y_bounds = sorted([roi[0][1], roi[1][1]])
                if x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1]:
                    self.parking_spot_classification[i] = not self.parking_spot_classification[i]
                    refresh = True
        if refresh:
            self._draw_classifications()


    def _draw_classifications(self):
        '''
        Draws the classications starting fresh
        '''
        self.image = self.clone.copy()
        for i, spot in enumerate(self.parking_spots):
            try:
                self.parking_spot_classification[i]
            except IndexError:
                self.parking_spot_classification.append(False)

            if self.parking_spot_classification[i]:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(self.image, spot[0], spot[1], color, 4)
        cv2.imshow(self.window_name, self.image)


    def _setup_windows(self):
        '''
        Runs our initial set up, open the window show, show the image and draw the parking spots
        '''
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(self.window_name, self.image)
        cv2.setMouseCallback(self.window_name, self._toggle_spot)
        self._draw_classifications()


    def start_loop(self):
        '''
        Loop until user hits "c"
        Reset parking spots on "r"
        '''
        self._setup_windows()
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("r") or key == ord("R"):
                self._reset_parking_spots()
            elif key == ord("c") or key == ord("C"):
                break
        self.parking_spot_classification.save()
        cv2.destroyAllWindows()


def classify_folder(folder, select_first=False):
    image_files = glob(folder + '/*.jpg')
    if select_first:
        ParkingSpotSelector(image_files[0]).start_loop()
    for f in image_files:
        ParkingSpotClassifier(f).start_loop()


class ParkingSpotPrediction(ParkingSpotClassifier):
    def _setup_windows(self):
        '''
        Runs our initial set up, open the window show, show the image and draw the parking spots
        '''
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self._draw_classifications()

    def _draw_classifications(self):
        '''
        Draws the classications starting
        '''
        self.image = self.clone.copy()
        for i, spot in enumerate(self.parking_spots):
            is_spot = models.ParkingSpotPredictor().predict(self.filename, spot, i)[0]

            if is_spot:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(self.image, spot[0], spot[1], color, 4)
        cv2.imshow(self.window_name, self.image)