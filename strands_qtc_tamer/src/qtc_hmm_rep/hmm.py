#!/usr/bin/env python

"""
(c) 2011, 2012 Georgia Tech Research Corporation
This source code is released under the New BSD license.  Please see
http://wiki.quantsoftware.org/index.php?title=QSTK_License
for license details.

This package includes code for representing and learning HMM's.

Most of the code in this package was derived from the descriptions provided in
'A Tutorial on Hidden Markov Models and Selected Applications in Speach
Recognition' by Lawence Rabiner.

Conventions:
The keyword argument elem_size will be passed in when creating numpy array
objects.
"""
#import math,random,sys
import numpy as np

def calcalpha(stateprior,transition,emission,observations,numstates,elem_size=np.longdouble):
    """
    Calculates 'alpha' the forward variable.

    The alpha variable is a numpy array indexed by time, then state (TxN).
    alpha[t][i] = the probability of being in state 'i' after observing the
    first t symbols.
    """
    alpha = np.zeros((len(observations),numstates),dtype=elem_size)
    for x in xrange(numstates):
        alpha[0][x] = stateprior[x]*emission[x][observations[0]]
    for t in xrange(1,len(observations)):
        for j in xrange(numstates):
            for i in xrange(numstates):
                alpha[t][j] += alpha[t-1][i]*transition[i][j]
            alpha[t][j] *= emission[j][observations[t]]
    return alpha

def forwardbackward(stateprior,transition,emission,observations,numstates,elem_size=np.longdouble):
    """
    Calculates the probability of a sequence given the HMM.
    """
    alpha = calcalpha(stateprior,transition,emission,observations,numstates,elem_size)
    return sum(alpha[-1])

def calcbeta(transition,emission,observations,numstates,elem_size=np.longdouble):
    """
    Calculates 'beta' the backward variable.

    The beta variable is a numpy array indexed by time, then state (TxN).
    beta[t][i] = the probability of being in state 'i' and then observing the
    symbols from t+1 to the end (T).
    """
    beta = np.zeros((len(observations),numstates),dtype=elem_size)
    for s in xrange(numstates):
        beta[len(observations)-1][s] = 1.
    for t in xrange(len(observations)-2,-1,-1):
        for i in xrange(numstates):
            for j in xrange(numstates):
                beta[t][i] += transition[i][j]*emission[j][observations[t+1]]*beta[t+1][j]
    return beta

def calcxi(stateprior,transition,emission,observations,numstates,alpha=None,beta=None,elem_size=np.longdouble):
    """
    Calculates 'xi', a joint probability from the 'alpha' and 'beta' variables.

    The xi variable is a numpy array indexed by time, state, and state (TxNxN).
    xi[t][i][j] = the probability of being in state 'i' at time 't', and 'j' at
    time 't+1' given the entire observation sequence.
    """
    if alpha is None:
        alpha = calcalpha(stateprior,transition,emission,observations,numstates,elem_size)
    if beta is None:
        beta = calcbeta(transition,emission,observations,numstates,elem_size)
    print 'alpha: ', alpha
    print 'beta: ', beta
    xi = np.zeros((len(observations),numstates,numstates),dtype=elem_size)
    for t in xrange(len(observations)-1):
        denom = 0.0
        for i in xrange(numstates):
            for j in xrange(numstates):
                thing = 1.0
                thing *= alpha[t][i]
                thing *= transition[i][j]
                thing *= emission[j][observations[t+1]]
                thing *= beta[t+1][j]
                denom += thing
        for i in xrange(numstates):
            for j in xrange(numstates):
                numer = 1.0
                numer *= alpha[t][i]
                numer *= transition[i][j]
                numer *= emission[j][observations[t+1]]
                numer *= beta[t+1][j]
                #print 'Debug: ', numer, denom
                xi[t][i][j] = numer/denom
    return xi

def calcgamma(xi,seqlen,numstates, elem_size=np.longdouble):
    """
    Calculates 'gamma' from xi.

    Gamma is a (TxN) numpy array, where gamma[t][i] = the probability of being
    in state 'i' at time 't' given the full observation sequence.
    """
    gamma = np.zeros((seqlen,numstates),dtype=elem_size)
    for t in xrange(seqlen):
        for i in xrange(numstates):
            gamma[t][i] = sum(xi[t][i])
    return gamma

def baumwelchstep(stateprior,transition,emission,observations,numstates,numsym,elem_size=np.longdouble,pseudo_trans=True):
    """
    Given an HMM model and a sequence of observations, computes the Baum-Welch
    update to the parameters using gamma and xi.
    """

    #E-step
    #TODO: do E-step over all sequences not just one.
    xi = []
    gamma = []
    for obs in observations:
        print obs
        xi.append(calcxi(stateprior,transition,emission,obs,numstates,elem_size=elem_size))
        gamma.append(calcgamma(xi[-1],len(obs),numstates,elem_size))
    xi = np.asarray(xi)
    gamma = np.asarray(gamma)

    return gamma, xi

    #M-step
    #TODO: do M-step over all sequences not just one at a time. Maybe mean ov E-step?!
    print 'gamma: ', gamma
    newprior = gamma[0]
    newtrans = np.zeros((numstates,numstates),dtype=elem_size)
    for i in xrange(numstates):
        for j in xrange(numstates):
            numer = 0.0
            denom = 1e-15 if pseudo_trans else 0.0
            for t in xrange(len(observations)-1):
                numer += xi[t][i][j]
                denom += gamma[t][i]
            #print 'Numer: ',xi[t][i][j],' Denom: ',gamma[t][i]
            newtrans[i][j] = numer/denom
    newemiss = np.zeros( (numstates,numsym) ,dtype=elem_size)
    for j in xrange(numstates):
        for k in xrange(numsym):
            numer = 0.0
            denom = 1e-15 if pseudo_trans else 0.0
            for t in xrange(len(observations)):
                if observations[t] == k:
                    numer += gamma[t][j]
                denom += gamma[t][j]
            newemiss[j][k] = numer/denom
    return newprior,newtrans,newemiss

class HMMLearner:
    """
    A class for modeling and learning HMMs.

    This class conveniently wraps the module level functions. Class objects hold 6
    data members:
    - num_states               number of hidden states in the HMM
    - num_symbols              number of possible symbols in the observation
                               sequence
    - precision                precision of the numpy.array elements (defaults to
                               longdouble)
    - prior                    The prior probability of starting in each state
                               (Nx1 array)
    - transition_matrix        The probability of transitioning between each state
                              (NxN matrix)
    - emission_matrix          The probability of each symbol in each state
                               (NxO matrix)
    You can set the 3 matrix parameters as you wish, but make sure the shape of
    the arrays matches num_states and num_symbols, as these are used internally

    Typical usage of this class is to create an HMM with a set number of states
    and external symbols, train the HMM using addEvidence(...), and then use
    the sequenceProb(...) method to see how well a specific sequence matches
    the trained HMM.
    """
    def __init__(self,num_states,num_symbols,init_type='uniform',precision=np.longdouble,transition_matrix=None,emission_matrix=None,prior=None):
        """
        Creates a new HMMLearner object with the given number of internal
        states, and external symbols.

        calls self.reset(init_type=init_type)
        """
        self.num_states = num_states
        self.num_symbols = num_symbols
        self.precision = precision
        self.predef_transision_matrix = transition_matrix
        self.predef_emission_matrix = emission_matrix
        self.predef_prior = prior
        self.reset(init_type=init_type)

    def reset(self, init_type='uniform'):
        """
        Resets the 3 arrays using the given initialization method.

        Wipes out the old arrays. You can use this method to change the shape
        of the arrays by first changing num_states and/or num_symbols, and then
        calling this method.

        Currently supported initialization methods:
        uniform        prior, transition, and emission probabilities are all
                    uniform (default)
        predefined      prior, transition, and emission have to be predefined
                        on initialisation.
        """
        if init_type == 'uniform':
            self.prior = np.ones( (self.num_states), dtype=self.precision) *(1.0/self.num_states)
            self.transition_matrix = np.ones( (self.num_states,self.num_states), dtype=self.precision)*(1.0/self.num_states)
            self.emission_matrix = np.ones( (self.num_states,self.num_symbols), dtype=self.precision)*(1.0/self.num_symbols)
        elif init_type == 'predefined':
            self.prior = self.predef_prior
            self.transition_matrix = self.predef_transision_matrix
            self.emission_matrix = self.predef_emission_matrix

    def sequenceProb(self, newData):
        """
        Returns the probability that this HMM generated the given sequence.

        Uses the forward-backward algorithm.  If given an array of
        sequences, returns a 1D array of probabilities.
        """
        if len(newData.shape) == 1:
            return forwardbackward( self.prior,\
                                    self.transition_matrix,\
                                    self.emission_matrix,\
                                    newData,\
                                    self.num_states,\
                                    self.precision)
        elif len(newData.shape) == 2:
            return np.array([forwardbackward(self.prior,self.transition_matrix,self.emission_matrix,newSeq,self.num_states,self.precision) for newSeq in newData])

    def printStatus(self,pdiff,tdiff,ediff,it):
        """Prints some status information while training"""
        print 'Iteration: ', it, ':'
        print 'Errors   : prior: ', pdiff, ' transition: ', tdiff, ' emission: ', ediff


    def addEvidence(self, newData, iterations=1,epsilon=0.0):
        """
        Updates this HMMs parameters given a new set of observed sequences
        using the Baum-Welch algorithm.

        newData can either be a single (1D) array of observed symbols, or a 2D
        matrix, each row of which is a seperate sequence. The Baum-Welch update
        is repeated 'iterations' times, or until the sum absolute change in
        each matrix is less than the given epsilon.  If given multiple
        sequences, each sequence is used to update the parameters in order, and
        the sum absolute change is calculated once after all the sequences are
        processed.
        """
        for i in xrange(iterations):
            newp,newt,newe = baumwelchstep( self.prior, \
                                            self.transition_matrix, \
                                            self.emission_matrix, \
                                            newData, \
                                            self.num_states, \
                                            self.num_symbols,\
                                            self.precision)
            pdiff = sum([abs(np-op) for np in newp for op in self.prior])
            tdiff = sum([sum([abs(nt-ot) for nt in newti for ot in oldt]) for newti in newt for oldt in self.transition_matrix])
            ediff = sum([sum([abs(ne-oe) for ne in newei for oe in olde]) for newei in newe for olde in self.emission_matrix])
            #self.printStatus(pdiff,tdiff,ediff,i)
            if(pdiff < epsilon) and (tdiff < epsilon) and (ediff < epsilon):
                break
            self.prior = newp
            self.transition_matrix = newt
            self.emission_matrix = newe

