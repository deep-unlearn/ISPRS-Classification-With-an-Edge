import numpy as np
import operator
import functools
#import pdb


"""
    This method computes the total number of parameters in a CAFFE fully convolutional network (FCN).

    Best USE: Stop training in debug mode and run the script

    

    === TODO ===

        * Modify it so that can also be used for fully-connected parts in a network

"""

def count_num_params(solver):

    # total parameter counter - initialize
    total_counter=0
        
    all_keys = solver.net.params.keys()
    
    for ii in range (len(all_keys)):
        
        # initialize
        this_layers_params_num=0

        this_layers_params_num = solver.net.params[all_keys[ii]][0].data.shape
        num_of_params=functools.reduce(operator.mul, this_layers_params_num, 1)

        try:
            num_of_params += solver.net.params[all_keys[ii]][1].data.shape[0]

        except IndexError:  # network does not have a bias parameter
            num_of_params += 0  

        # sum all parameters of all layers
        total_counter += num_of_params

    return total_counter
