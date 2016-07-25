# Parking-Spot-Detector
Automatically Detect Parking Spots on your block!

## Step One: 
Run main.py with the argument select to select your parking spaces:

``` python main.py select --file=image1.jpg ```

## Step Two:
Run main.py with the argument classify to open a folder of images and mark whether your preselected parking spots are occupied.

``` python main.py classify --folder=scene ```

## Step Three:
Now that you have parking spots selected and classified, run train to train and persist your classifier. 

``` python main.py train```

## Step Four:

Make predictions on files or whole folders. 

``` python main.py predict --file=image2.jpg```



