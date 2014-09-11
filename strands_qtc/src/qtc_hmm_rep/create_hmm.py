#!/usr/bin/env python

import numpy as np
import ghmm as gh
import os


class QtcException(Exception):
    def __init__(self, message):

        # Call the base class constructor with the parameters it
        # needs
        Exception.__init__(self, "QTC Exception: " + message)


def readQtcFiles(path):
    """reads all .qtc files from a given directory and resturns them as numpy arrays"""

    qtc = []
    for f in os.listdir(path):
        if f.endswith(".qtc"):
            filename = path + '/' + f
            qtc.append(np.genfromtxt(filename, delimiter=','))
    return qtc


def createSequenceSet(qtc, symbols):
    return gh.SequenceSet(symbols, qtc)


def createCNDTransEmiProb(qtc_type='qtcc'):
    """Creates a Conditional Neighbourhood Diagram as a basis for the HMM"""

    if qtc_type is 'qtcb':
        state_num = 11
    elif qtc_type is 'qtcc':
        state_num = 83
    elif qtc_type is 'qtcbc':
        state_num = 92
    else:
        raise(QtcException("Unknow qtc type: {!r}".format(qtc_type)))

    qtc = []

    if qtc_type is 'qtcb':
        for i in xrange(1, 4):
            for j in xrange(1, 4):
                qtc.append([i-2, j-2])
    elif qtc_type is 'qtcc':
        for i in xrange(1, 4):
            for j in xrange(1, 4):
                for k in xrange(1, 4):
                    for l in xrange(1, 4):
                        qtc.append([i-2, j-2, k-2, l-2])
    elif qtc_type is 'qtcbc':
        for i in xrange(1, 4):
            for j in xrange(1, 4):
                qtc.append([i-2, j-2, np.NaN, np.NaN])
        for i in xrange(1, 4):
            for j in xrange(1, 4):
                for k in xrange(1, 4):
                    for l in xrange(1, 4):
                        qtc.append([i-2, j-2, k-2, l-2])
    else:
        raise(QtcException("Unknow qtc type: {!r}".format(qtc_type)))

    qtc = np.array(qtc)

    trans = np.zeros((state_num, state_num))
    for i1 in xrange(qtc.shape[0]):
        for i2 in xrange(i1+1, qtc.shape[0]):
            trans[i1+1, i2+1] = np.absolute(qtc[i1]-qtc[i2]).max() != 2
            if trans[i1+1, i2+1] == 1:
                for j1 in xrange(qtc.shape[1]-1):
                    for j2 in xrange(j1+2, qtc.shape[1]+1):
                        if sum(np.absolute(qtc[i1, j1:j2])) == 1 \
                                and sum(np.absolute(qtc[i2, j1:j2])) == 1:
                            if max(np.absolute(qtc[i1, j1:j2]-qtc[i2, j1:j2])) > 0 \
                                    and sum(qtc[i1, j1:j2]-qtc[i2, j1:j2]) != 1:
                                trans[i1+1, i2+1] = 5
                                break
                if trans[i1+1, i2+1] != 1:
                    break
            trans[i2+1, i1+1] = trans[i1+1, i2+1]

    trans[trans != 1] = 0
    np.savetxt('/home/cdondrup/trans.csv', trans, delimiter=',', fmt='%1f')
    trans[0] = 1
    trans[:, 0] = 0
    trans[:, -1] = 1
    trans[0, -1] = 0
    trans[-1] = 0
    trans += np.dot(np.eye(state_num), 0.00001)
    trans[0, 0] = 0

    #trans[trans == 0] = 0.00001

    trans = trans / trans.sum(axis=1).reshape(-1, 1)
    #np.savetxt('/home/cdondrup/trans.csv', trans, delimiter=',')

    emi = np.eye(state_num)
    emi[emi == 0] = 0.0001

    return trans, emi


def qtc2state(qtc):
    """Transforms a qtc state to a number"""

    state_rep = []
    for idx, element in enumerate(qtc):
        val_qtc = validateQtcSequences(element)
        d = val_qtc.shape[1]
        mult = 3**np.arange(d-1, -1, -1)
        state_num = np.append(
            0,
            ((val_qtc + 1)*np.tile(mult, (val_qtc.shape[0], 1))).sum(axis=1) + 1
        )
        state_num = np.append(state_num, 82)
        state_char = ''
        for n in state_num:
            state_char += chr(int(n)+32)
        state_rep.append(state_num.tolist())

    return state_rep


def validateQtcSequences(qtc):
    """Removes illegal state transition by inserting necessary intermediate states"""

    newqtc = qtc[0].copy()
    j = 1
    for i in xrange(1, len(qtc)):
        checksum = np.nonzero(qtc[i-1]+qtc[i] == 0)
        intermediate = qtc[i].copy()
        if checksum[0].size > 0:
            for idx in checksum[0]:
                if np.absolute(qtc[i-1][idx]) + np.absolute(qtc[i][idx]) > 0:
                    intermediate[idx] = 0
            if np.any(intermediate != qtc[i]):
                newqtc = np.append(newqtc, intermediate)
                j += 1
        newqtc = np.append(newqtc, qtc[i])
        j += 1

    return newqtc.reshape(-1, 4)


def generateAlphabet(num_symbols):
    return gh.IntegerRange(0, num_symbols)


def trainHMM(seq, trans, emi, qtc_type='qtcc'):
    """Uses the given parameters to train a multinominal HMM to represent the given seqences"""

    if qtc_type is 'qtcb':
        state_num = 11
    elif qtc_type is 'qtcc':
        state_num = 83
    elif qtc_type is 'qtcbc':
        state_num = 92
    else:
        raise(QtcException("Unknow qtc type: {!r}".format(qtc_type)))

    print 'Generating HMM:'
    print '\tCreating symbols...'
    symbols = generateAlphabet(state_num)
    startprob = np.zeros((state_num))
    startprob[0] = 1
    print '\t\t', symbols
    print '\tCreating HMM...'
    qtc_hmm = gh.HMMFromMatrices(
        symbols,
        gh.DiscreteDistribution(symbols),
        trans.tolist(),
        emi.tolist(),
        startprob.tolist()
    )
    print '\tTraining...'
    qtc_hmm.baumWelch(createSequenceSet(seq, symbols))

    return qtc_hmm


def createHMM(seq_path, qtc_type='qtcc'):
    """Create and trains a HMM to represent the given qtc sequences"""

    try:
        qtc_seq = readQtcFiles(seq_path)
        qtc_state_seq = qtc2state(qtc_seq)
        trans, emi = createCNDTransEmiProb(qtc_type)
        qtchmm = trainHMM(qtc_state_seq, trans, emi, qtc_type)
        print '...done'
        return qtchmm
    except QtcException as e:
        print e.message


def createTestSequence(seq_path, qtc_type='qtcc'):
    if qtc_type is 'qtcb':
        return createSequenceSet(qtc2state(readQtcFiles(seq_path)), generateAlphabet(11))
    elif qtc_type is 'qtcc':
        return createSequenceSet(qtc2state(readQtcFiles(seq_path)), generateAlphabet(83))
    elif qtc_type is 'qtcbc':
        return createSequenceSet(qtc2state(readQtcFiles(seq_path)), generateAlphabet(92))
    else:
        raise(QtcException("Unknow qtc type: {!r}".format(qtc_type)))