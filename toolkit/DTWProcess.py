from fastdtw import fastdtw as dtw
import numpy as np
import pandas as pd
import multiprocessing
import os
class PatternRecognition:
    def __init__(self, pattern, stepsize):
        self.PAT = pattern
        self.steps = stepsize

    def _fit(self, seq):
        windowsizes = [step for step in range((len(self.PAT)-self.steps), (len(self.PAT) + self.steps))]
        self.Results, self.Scores = list(), list()
        # works assignment
        works = [[seq, step, self.PAT] for step in windowsizes]
        # multiprocessing allocation
        Pools = multiprocessing.Pool(16)
        Temp = Pools.map(self.DTW_Compute, works)
        Scores = [alist[1] for alist in Temp]
        Results = [alist[0] for alist in Temp]
        idxs = np.argsort(Scores)
        self.Scores = [Scores[idx] for idx in idxs]
        self.Results = [Results[idx] for idx in idxs]
    
    def DTW_Compute(self, alist):
        SEQ, step, PAT = alist
        for i in range(len(SEQ) - step):
            TEMP_SEQ = SEQ[i:i+step]
            dist, path = dtw(TEMP_SEQ, PAT)
            return [[i, i+step], dist/len(path)]
