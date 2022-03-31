import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression
import json


class confidence_prediction:
    '''
    Features: [Max_l, Avg_l, logFreq, MissingCount, Pace]
    Pace = characters per second
    MissingCount = number of OOV words in frequency dict normalized by total number of words
    Max_l, Avg_l = max and average normalised levenshtein distance between predictions (w/ dropout) and reference (w/o dropout)
    mean_freq = mean frequency of all words in the validation set
    '''
    def __init__(
        self, 
        freq_path='sfreqs.json',
        coef=[0.23435166, 1.71177441, -0.04422353, 0.25170083, -0.00191587],
        interp=0.3652430901674939,
        mean_freq=5.865553144322172
    ):
        self.coef = np.array(coef)
        self.interp = np.array(interp)
        self.sfreq = json.load(open(freq_path))
        self.mean_freq = mean_freq
        self.model = LinearRegression()
        self.model.coef_ = self.coef
        self.model.intercept_ = self.interp

    @staticmethod
    def __get_pace(text, length):
        return len(text)/length

    def __get_freq_data(self, text):
        words = text.lower().strip().split(' ')
        wordfreqs = [self.sfreq[word] if word in self.sfreq else None for word in words]
        misses = sum(1 for word in wordfreqs if word == None) / len(words)
        wordfreqs = [el for el in wordfreqs if el != None]
        sfreqsn = sum(wordfreqs)/len(wordfreqs) if len(wordfreqs) != 0 else self.mean_freq
        return sfreqsn, misses


    @staticmethod
    def __build_features(max_l, avg_l, logFreq, missingCount, pace):
        return np.array([max_l, avg_l, logFreq, missingCount, pace]).reshape(1,-1)

    def predict(self, max_l, avg_l, text, length):
        pace = self.__get_pace(text, length)
        wordfreqs, misses = self.__get_freq_data(text)
        feat = self.__build_features(max_l, avg_l, wordfreqs, misses, pace)
        return self.model.predict(feat).item()