"""
Copyright (C) 2014 Dallas Card

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.


Description:
Given unreliable observations of patient classes by multiple observers,
determine the most likely true class for each patient, class marginals,
and  individual error rates for each observer, using Expectation Maximization


References:
( Dawid and Skene (1979). Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm. Journal of the Royal Statistical Society.
Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28. 
"""

import numpy as np
import sys

"""
Function: main()
    Run the EM estimator on the data from the Dawid-Skene paper
"""
def main():
    # load the data from the paper
    responses = generate_sample_data()
    # run EM
    run(responses)

"""
Function: dawid_skene()
    Run the Dawid-Skene estimator on response data
Input:
    responses: a dictionary object of responses:
        {patients: {observers: [labels]}}
    tol: tolerance required for convergence of EM
    max_iter: maximum number of iterations of EM
""" 
def run(responses, tol=0.00001, max_iter=100, init='average'):
    # convert responses to counts
    (patients, observers, classes, counts) = responses_to_counts(responses)
    print "num Patients:", len(patients)
    print "Observers:", observers
    print "Classes:", classes
    
    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None

    patient_classes = initialize(counts)
    
    print "Iter\tlog-likelihood\tdelta-CM\tdelta-ER"    
    
    # while not converged do:
    while not converged:     
        iter += 1
        
        # M-step
        (class_marginals, error_rates) = m_step(counts, patient_classes)        
 
        # E-setp
        patient_classes = e_step(counts, class_marginals, error_rates)  
        
        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)
        
        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
            print iter ,'\t', log_L, '\t%.6f\t%.6f' % (class_marginals_diff, error_rates_diff)            
            if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                converged = True
        else:
            print iter ,'\t', log_L
    
        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates
                
    # Print final results
    np.set_printoptions(precision=2, suppress=True)
    print "Class marginals"
    print class_marginals
    print "Error rates"
    print error_rates

    print "Incidence-of-error rates"
    [nPatients, nObservers, nClasses] = np.shape(counts)
    for k in range(nObservers):
        print class_marginals * error_rates[k,:,:]

    np.set_printoptions(precision=4, suppress=True)    
    print "Patient classes"
    for i in range(nPatients):
        print patients[i], patient_classes[i,:] 
        
    #return (patients, observers, classes, counts, class_marginals, error_rates, patient_classes) 
 
"""
Function: responses_to_counts()
    Convert a matrix of annotations to count data
Inputs:
    responses: dictionary of responses {patient:{observers:[responses]}}
Return:
    patients: list of patients
    observers: list of observers
    classes: list of possible patient classes
    counts: 3d array of counts: [patients x observers x classes]
""" 
def responses_to_counts(responses):
    patients = responses.keys()
    patients.sort()
    nPatients = len(patients)
        
    # determine the observers and classes
    observers = set()
    classes = set()
    for i in patients:
        i_observers = responses[i].keys()
        for k in i_observers:
            if k not in observers:
                observers.add(k)
            ik_responses = responses[i][k]
            classes.update(ik_responses)
    
    classes = list(classes)
    classes.sort()
    nClasses = len(classes)
        
    observers = list(observers)
    observers.sort()
    nObservers = len(observers)
            
    # create a 3d array to hold counts
    counts = np.zeros([nPatients, nObservers, nClasses])
    
    # convert responses to counts
    for patient in patients:
        i = patients.index(patient)
        for observer in responses[patient].keys():
            k = observers.index(observer)
            for response in responses[patient][observer]:
                j = classes.index(response)
                counts[i,k,j] += 1
        
    
    return (patients, observers, classes, counts)


"""
Function: initialize()
    Get initial estimates for the true patient classes using counts
    see equation 3.1 in Dawid-Skene (1979)
Input:
    counts: counts of the number of times each response was received 
        by each observer from each patient: [patients x observers x classes] 
Returns:
    patient_classes: matrix of estimates of true patient classes:
        [patients x responses]
"""  
def initialize(counts):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)
    # create an empty array
    patient_classes = np.zeros([nPatients, nClasses])
    # for each patient, take the average number of observations in each class
    for p in range(nPatients):
        patient_classes[p,:] = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        
    return patient_classes


"""
Function: m_step()
    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true patient classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979)
Input: 
    counts: Array of how many times each response was received
        by each observer from each patient
    patient_classes: Matrix of current assignments of patients to classes
Returns:
    p_j: class marginals [classes]
    pi_kjl: error rates - the probability of observer k receiving
        response l from a patient in class j [observers, classes, classes]
"""
def m_step(counts, patient_classes):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    
    # compute class marginals
    class_marginals = np.sum(patient_classes,0)/float(nPatients)
    
    # compute error rates 
    error_rates = np.zeros([nObservers, nClasses, nClasses])
    for k in range(nObservers):
        for j in range(nClasses):
            for l in range(nClasses): 
                error_rates[k, j, l] = np.dot(patient_classes[:,j], counts[:,k,l])
            # normalize by summing over all observation classes
            sum_over_responses = np.sum(error_rates[k,j,:])
            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:]/float(sum_over_responses)  

    return (class_marginals, error_rates)


""" 
Function: e_step()
    Determine the probability of each patient belonging to each class,
    given current ML estimates of the parameters from the M-step
    See equation 2.5 in Dawid-Skene (1979)
Inputs:
    counts: Array of how many times each response was received
        by each observer from each patient
    class_marginals: probability of a random patient belonging to each class
    error_rates: probability of observer k assigning a patient in class j 
        to class l [observers, classes, classes]
Returns:
    patient_classes: Soft assignments of patients to classes
        [patients x classes]
"""      
def e_step(counts, class_marginals, error_rates):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    
    patient_classes = np.zeros([nPatients, nClasses])    
    
    for i in range(nPatients):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            
            patient_classes[i,j] = estimate
        # normalize error rates by dividing by the sum over all observation classes
        patient_sum = np.sum(patient_classes[i,:])
        if patient_sum > 0:
            patient_classes[i,:] = patient_classes[i,:]/float(patient_sum)
    
    return patient_classes


"""
Function: calc_likelihood()
    Calculate the likelihood given the current parameter estimates
    This should go up monotonically as EM proceeds
    See equation 2.7 in Dawid-Skene (1979)
Inputs:
    counts: Array of how many times each response was received
        by each observer from each patient
    class_marginals: probability of a random patient belonging to each class
    error_rates: probability of observer k assigning a patient in class j 
        to class l [observers, classes, classes]
Returns:
    Likelihood given current parameter estimates
"""  
def calc_likelihood(counts, class_marginals, error_rates):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    log_L = 0.0
    
    for i in range(nPatients):
        patient_likelihood = 0.0
        for j in range(nClasses):
        
            class_prior = class_marginals[j]
            patient_class_likelihood = np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))  
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior
                              
        temp = log_L + np.log(patient_likelihood)
        
        if np.isnan(temp) or np.isinf(temp):
            print i, log_L, np.log(patient_likelihood), temp
            sys.exit()

        log_L = temp        
        
    return log_L
    

"""
Function: generate_sample_data()
    Generate the data from Table 1 in Dawid-Skene (1979) in the proper format
"""  
def generate_sample_data():
    responses = {
                 1: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 2: {1:[3,3,3], 2:[4], 3:[3], 4:[3], 5:[4]},
                 3: {1:[1,1,2], 2:[2], 3:[1], 4:[2], 5:[2]},
                 4: {1:[2,2,2], 2:[3], 3:[1], 4:[2], 5:[1]},
                 5: {1:[2,2,2], 2:[3], 3:[2], 4:[2], 5:[2]},
                 6: {1:[2,2,2], 2:[3], 3:[3], 4:[2], 5:[2]},
                 7: {1:[1,2,2], 2:[2], 3:[1], 4:[1], 5:[1]},
                 8: {1:[3,3,3], 2:[3], 3:[4], 4:[3], 5:[3]},
                 9: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[3]},
                 10: {1:[2,3,2], 2:[2], 3:[2], 4:[2], 5:[3]},
                 11: {1:[4,4,4], 2:[4], 3:[4], 4:[4], 5:[4]},
                 12: {1:[2,2,2], 2:[3], 3:[3], 4:[4], 5:[3]},
                 13: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 14: {1:[2,2,2], 2:[3], 3:[2], 4:[1], 5:[2]},
                 15: {1:[1,2,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 16: {1:[1,1,1], 2:[2], 3:[1], 4:[1], 5:[1]},
                 17: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 18: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 19: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[1]},
                 20: {1:[2,2,2], 2:[1], 3:[3], 4:[2], 5:[2]},
                 21: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 22: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[1]},
                 23: {1:[2,2,2], 2:[3], 3:[2], 4:[2], 5:[2]},
                 24: {1:[2,2,1], 2:[2], 3:[2], 4:[2], 5:[2]},
                 25: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 26: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 27: {1:[2,3,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 28: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 29: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 30: {1:[1,1,2], 2:[1], 3:[1], 4:[2], 5:[1]},
                 31: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 32: {1:[3,3,3], 2:[3], 3:[2], 4:[3], 5:[3]},
                 33: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 34: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 35: {1:[2,2,2], 2:[3], 3:[2], 4:[3], 5:[2]},
                 36: {1:[4,3,3], 2:[4], 3:[3], 4:[4], 5:[3]},
                 37: {1:[2,2,1], 2:[2], 3:[2], 4:[3], 5:[2]},
                 38: {1:[2,3,2], 2:[3], 3:[2], 4:[3], 5:[3]},
                 39: {1:[3,3,3], 2:[3], 3:[4], 4:[3], 5:[2]},
                 40: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 41: {1:[1,1,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 42: {1:[1,2,1], 2:[2], 3:[1], 4:[1], 5:[1]},
                 43: {1:[2,3,2], 2:[2], 3:[2], 4:[2], 5:[2]},
                 44: {1:[1,2,1], 2:[1], 3:[1], 4:[1], 5:[1]},
                 45: {1:[2,2,2], 2:[2], 3:[2], 4:[2], 5:[2]}
                 }
    return responses


"""
Function: random_initialization()
    Alternative initialization # 1
    Similar to initialize() above, except choose one initial class for each
    patient, weighted in proportion to the counts
Input:
    counts: counts of the number of times each response was received 
        by each observer from each patient: [patients x observers x classes] 
Returns:
    patient_classes: matrix of estimates of true patient classes:
        [patients x responses]
"""  
def random_initialization(counts):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    
    response_sums = np.sum(counts,1)
    
    # create an empty array
    patient_classes = np.zeros([nPatients, nClasses])
    
    # for each patient, choose a random initial class, weighted in proportion
    # to the counts from all observers
    for p in range(nPatients):
        average = response_sums[p,:] / np.sum(response_sums[p,:],dtype=float)
        patient_classes[p,np.random.choice(np.arange(nClasses), p=average)] = 1
        
    return patient_classes


"""
Function: majority_voting()
    Alternative initialization # 2
    An alternative way to initialize assignment of patients to classes 
    i.e Get initial estimates for the true patient classes using majority voting
    This is not in the original paper, but could be considered
Input:
    counts: Counts of the number of times each response was received 
        by each observer from each patient: [patients x observers x classes] 
Returns:
    patient_classes: matrix of initial estimates of true patient classes:
        [patients x responses]
"""  
def majority_voting(counts):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)
    
    # create an empty array
    patient_classes = np.zeros([nPatients, nClasses])
    
    # take the most frequent class for each patient 
    for p in range(nPatients):        
        indices = np.argwhere(response_sums[p,:] == np.max(response_sums[p,:]))
        # in the case of ties, take the lowest valued label (could be randomized)        
        patient_classes[p, np.min(indices)] = 1
        
    return patient_classes


if __name__ == '__main__':
    main()
