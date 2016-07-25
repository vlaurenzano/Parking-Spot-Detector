import gui
import argparse
import models

if __name__ != '__main__':
    exit('This is not intended for impocrt...')

parser = argparse.ArgumentParser(description="Detect open parking spots by using the modules to select the spaces in a scene")
parser.add_argument('module', help='the module to run, select, classify, predict')
parser.add_argument('--file',   help='the file to use for selecting spaces')
parser.add_argument('--folder',   help='the folder to iterate over for training or monitoring')
args = parser.parse_args()


if args.module == 'select':
    if not args.file:
        exit('Need to supply an image to select spaces')
    gui.ParkingSpotSelector(args.file).start_loop()
elif args.module == 'classify':
    if not args.folder:
       exit('Need to supply a folder to classify')
    gui.classify_folder(args.file_folder)
elif args.module == 'test':
   if not args.folder:
       exit('Need to supply a folder')
   psp = models.ParkingSpotPredictor()
   psp.fit(args.folder, model=True)
elif args.module == 'train':
   if not args.folder:
    exit('Need to supply a folder')
   psp = models.ParkingSpotPredictor()
   psp.fit(args.folder, model=False)
   print 'Model trained, run predict or monitor to use'
elif args.module == 'predict':
    if args.file:
        gui.ParkingSpotPrediction(args.file).start_loop()
    elif args.folder:
        from glob import glob
        image_files = glob(args.folder + '/*.jpg')
        for f in image_files:
            gui.ParkingSpotPrediction(f).start_loop()
    else:
        exit('Need to supply an image or folder to predict')
