from __future__ import division
import os
os.environ['GLOG_minloglevel'] = '0'
from matplotlib import pyplot as plt
from visualizations import vis
import numpy as np
import caffe
import pdb
import signal

# enable for explicit debugging
#pdb.set_trace()

############  Signal Handler - Real Time Manipulation of CNN ##############

def signal_handler(signal_number, frame):
    pdb.set_trace()

# ##########################  HYPERPARAMETERS ################################ #

def main(nsteps, print_out_interval, test_interval, num_test_iter):

    # variable initalization 
    train_loss = np.zeros(nsteps)
    test_acc = np.zeros(int(np.ceil(nsteps / test_interval)))
    batch_error = []
    edge_error = []

    # take gradient steps
    for it in range(nsteps):

        # when ignite singnal function
        signal.signal(signal.SIGINT, signal_handler)

        # take a single step SGD step
        solver.step(1)

        # store the all SGD train loss
        #train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='edge_dem_conv1_1_')
        #output[it] = solver.test_nets[0].blobs['loss'].data[:8]

        # print-out interval
        if (it % print_out_interval == 0):

            print "------------------------------------------------------------------------------------"
            print "ITERATION : ",  it
            print "------------------------------------------------------------------------------------"
            print " FCN loss   : ", solver.net.blobs['loss-fcn'].data
            print " DS-5 loss  : ", solver.net.blobs['ds_loss5'].data
            print " DS-4 loss  : ", solver.net.blobs['ds_loss4'].data
            print " DS-3 loss  : ", solver.net.blobs['ds_loss3'].data
            print " DS-2 loss  : ", solver.net.blobs['ds_loss2'].data
            print " ====================================================="
            print " "
            print " Fuse loss  : ", solver.net.blobs['loss'].data
            print " Edges loss : ", solver.net.blobs['edge-loss'].data
            print " "        
            print " ====================================================="

            # store batch error for learning-trend analysis
            batch_error.append(np.array(solver.net.blobs['loss'].data))  
            edge_error.append(np.array(solver.net.blobs['edge-loss'].data))  

            # run evaluation over validation-set
            if (it % test_interval == 0) and (it!=0):
                print "------------------------------------------------------------------------------------"
                print '####################################################################################'
                print '####################################################################################'
                print "------------------------------------------------------------------------------------"

                # store values for computing statistics
                batch_size = solver.test_nets[0].blobs['label'].data.shape[0]
                num_pixels_prediction_instance = solver.test_nets[0].blobs['label'].data.shape[2] * solver.test_nets[0].blobs['label'].data.shape[3]

                # Copy all weights from the TRAIN to the TEST NETWORK
                all_layers = [k for k in solver.net.params.keys()]
                for ll in all_layers:

                    # try to copy WEIGHTS & BIASES if exist for layer 
                    try:
                        solver.test_nets[0].params[ll][0].data[:, :, :, :] = solver.net.params[ll][0].data[:, :, :, :]
                        solver.test_nets[0].params[ll][1].data[:] = solver.net.params[ll][1].data[:]

                    except IndexError:
                        # do not copy biases if dont exist. Copy only weights
                        solver.test_nets[0].params[ll][0].data[:, :, :, :] = solver.net.params[ll][0].data[:, :, :, :]

                # initialize
                correct = 0
                for test_it in range(num_test_iter):
                    solver.test_nets[0].forward()

                    # detect all correct predictions for this batch of images
                    temp_correct = solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data[:, 0, :, :]

                    # store their sum
                    temp_correct = np.float32(temp_correct.sum())

                    # accumulate to total sum
                    correct += temp_correct

                    # clear
                    temp_correct = 0

                testset_correct_prc = correct / (num_test_iter * batch_size * num_pixels_prediction_instance)

                # clear
                correct = 0
                
                print "------------------------------------------------------------------------------------"
                print '####################################################################################'
                print '####################################################################################'

                print 'VALIDATION SET ACCURACY : ', testset_correct_prc
                print "------------------------------------------------------------------------------------"
                print '####################################################################################'
                print '####################################################################################'

                # store validation set accuracy per iteration
                test_acc[it // test_interval] = testset_correct_prc

                # clear
                testset_correct_prc = 0

# ============================================================ #


if __name__ == "__main__":

# --------------- INPUTS -------------- #

    # select gpu
    idx_gpu = 1

    # set pre-trained weights for model
    base_weights = 'out_models/__best_model/_train_on_validation_and_training/train_iter_57000.caffemodel'

    # set solver file
    solver = 'solver.prototxt'

    # total number of training SGD steps
    nsteps = 60000

    # print error every "x" number of SGD steps
    print_out_interval = 10

    # test-network every "x" numver of SGD steps
    test_interval = 1000

    # number of batches to be tested in TEST-phase
    num_test_iter = 2000 

# --------------------------------------#

    # set gpu
    caffe.set_mode_gpu()
    caffe.set_device(idx_gpu)

    # initialize solver
    solver = caffe.SGDSolver(solver)

    # copy base weights for architecture
    solver.net.copy_from(base_weights)
    
    # call main function for training
    main(nsteps, print_out_interval, test_interval, num_test_iter)
