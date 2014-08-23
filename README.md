
## dawid_skene

This project is an implementation of the estimator for combining unreliable observations from Dawid and Skene (1979).

Given unreliable observations of patient classes by multiple observers,
determine the most likely true class for each patient, class marginals,
and  individual error rates for each observer, using Expectation Maximization

The example from the paper can be executed by running this script from the command line, i.e.
> python dawid_skene.py

Alternatively, it can be run within a python script as follows:

> import dawid_skene
> 
> responses = dawid_skene.generate_sample_data()
> 
> dawid_skene.run(responses)


### References:

* Dawid and Skene (1979). [Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm](http://www.cs.mcgill.ca/~jeromew/comp766/samples/Output_aggregation.pdf). Journal of the Royal Statistical Society.
Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28. 

