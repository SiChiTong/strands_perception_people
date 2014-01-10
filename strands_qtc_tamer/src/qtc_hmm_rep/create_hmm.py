#!/usr/bin/env python

import numpy as np
#import hmm
import ghmm as gh
import os

def readQtcFiles(path):
    """reads all .qtc files from a given directory and resturns them as numpy arrays"""

    qtc = []
    for f in os.listdir(path):
        if f.endswith(".qtc"):
            filename = path + '/' + f
            qtc.append(np.genfromtxt(filename,delimiter=','))
    return qtc


def createCNDTransEmiProb():
    """Creates a Conditional Neighbourhood Diagram as a basis for the HMM"""

    qtc = np.zeros((81,4))

    n = 0;
    for i in range(1,4):
        for j in range(1,4):
            for k in range(1,4):
                for l in range(1,4):
                    qtc[n] = [i-2,j-2,k-2,l-2]
                    n+=1;

    trans = np.zeros((83,83))
    for i in range(qtc.shape[0]):
        for j in range(qtc.shape[0]):
             trans[i+1,j+1] = np.absolute(qtc[i]-qtc[j]).max()

    trans[trans!=1] = 0
    trans[trans==0] = 1
    trans[0]    = 1
    trans[:,0]  = 0
    trans[:,-1] = 1
    trans[0,-1] = 0
    trans[-1]   = 0
    trans += np.dot(np.eye(83),0.00001)
    trans[0,0] = 0
    trans = trans / trans.sum(axis=1).reshape(-1,1)

    emi = np.eye(83)
    print trans
    return trans, emi

def qtc2state(qtc):
    """Transforms a qtc state to a number"""

    state_rep = []
    for idx, element in enumerate(qtc):
        val_qtc = validateQtcSequences(element)
        d = val_qtc.shape[1]
        mult = 3**np.arange(d-1,-1,-1)
        state_num = np.append(((val_qtc+1)*np.tile(mult,(val_qtc.shape[0],1))).sum(axis=1)+1,82)
        state_char = ''
        for n in state_num: state_char += chr(int(n)+32)
        #state_num += 1
        state_rep.append(state_num.tolist())

    return state_rep


def validateQtcSequences(qtc):
    """Removes illegal state transition by inserting necessary intermediate states"""

    newqtc = qtc[0].copy()
    j = 1
    for i in xrange(1,len(qtc)):
        checksum = np.nonzero(qtc[i-1]+qtc[i] == 0)
        intermediate = qtc[i].copy()
        if checksum[0].size > 0:
            for idx in checksum[0]:
                if np.absolute(qtc[i-1][idx]) + np.absolute(qtc[i][idx]) > 0:
                    intermediate[idx] = 0
            if np.any(intermediate != qtc[i]):
                newqtc = np.append(newqtc,intermediate)
                j += 1
        newqtc = np.append(newqtc,qtc[i])
        j+=1

    return newqtc.reshape(-1,4)

def trainHMM(seq, trans, emi, startprob):
    """Uses the given parameters to train a multinominal HMM to represent the given seqences"""

    symbols = gh.IntegerRange(0,83)
    print symbols
    qtc_hmm = gh.HMMFromMatrices(symbols, gh.DiscreteDistribution(symbols),trans.tolist(),emi.tolist(),startprob.tolist())
    #qtc_hmm = hmm.HMMLearner(83,83,init_type='predefined',transition_matrix=trans,emission_matrix=emi,prior=startprob)

    return qtc_hmm

def createHMM(seq_path):
    """Create and trains a HMM to represent the given qtc sequences"""

    qtc_seq = readQtcFiles(seq_path)
    qtc_state_seq = qtc2state(qtc_seq)
    trans, emi = createCNDTransEmiProb()
    startprob = np.zeros((83))
    startprob[41] = 1
    qtchmm = trainHMM(qtc_state_seq,trans,emi,startprob)
    return qtchmm
