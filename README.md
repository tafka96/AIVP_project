# AIVP_project
Check out the requirements.txt for the required libraries.
The images are provided in data.

#Vegetation Mask
The code for vegetation masks are in files histogram.py, k_means.py and NDVI.py for the 3 approaches.

#Classification
The classification Trainer should be run from command line:

python3 trainer.py

Arguments can be provided, for the best achieved results: 

python3 trainer.py --NDVI --epochs 40 --pretrainData coco --backbone resnet152 --lr 0.01

The model save training and test losses to files trainLosses.npy and testLosses.npy. The test confusion matrices will be saved in ConfusionMatrices.npy. After each epoch the model parameters are saved to two files: Model-symbol.json and Model-'epoch'.params where epoch stands for the epoch number. 
