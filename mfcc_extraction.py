from python_speech_features import mfcc
from python_speech_features import logfbank
from python_speech_features import delta
import scipy.io.wavfile as wav
import numpy
import os

# directory where we your .wav files are
directoryName = "dataset/senang/" # put your own directory here
# directory to put our results in, you can change the name if you like
resultsDirectory = directoryName + "/MFCCresults"

# make a new folder in this directory to save our results in
if not os.path.exists(resultsDirectory):
    os.makedirs(resultsDirectory)

# get MFCCs for every .wav file in our specified directory
for filename in os.listdir(directoryName):
    if filename.endswith('.wav'): # only get MFCCs from .wavs
        # read in our file
        (rate,sig) = wav.read(directoryName + "/" +filename)

        # get mfcc
        mfcc_feat = mfcc(sig,rate)
        d_mfcc_feat = delta(mfcc_feat, 2)
        # get filterbank energies
        # fbank_feat = logfbank(sig,rate)

        # create a file to save our results in
        outputFile = resultsDirectory + "/" + os.path.splitext(filename)[0] + ".csv"
        file = open(outputFile, 'w+') # make file/over write existing file
        numpy.savetxt(file, d_mfcc_feat, delimiter=",") #save MFCCs as .csv
        file.close() # close file