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

"""
   This method work along the mutltiscale (3x scales) SegNet model - 
   using two parallel streams for processing DEM data and IMAGES respectively 

   The prototxt for this model is very large hence the multiple printing outputs

"""


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


        solver.step(1)  # SGD by Caffe

        # store the train loss
        train_loss[it] = solver.net.blobs['loss'].data

        # store the output on the first test batch
        # (start the forward pass at conv1 to avoid loading new data)
        solver.test_nets[0].forward(start='edge_dem_conv1_1_')
        #output[it] = solver.test_nets[0].blobs['loss'].data[:8]

        if it % print_out_interval == 0:

            print "------------------------------------------------------------------------------------"
            print "ITERATION : ",  it
            print "------------------------------------------------------------------------------------"

            print " === edges loss ==> ", solver.net.blobs['edge-loss'].data
            print " loss-sc1    : ", solver.net.blobs['loss1'].data
            print '======================================================='
            #print " loss-sc2    : ", solver.net.blobs['loss2'].data 
            print " loss-sc2-up : ", solver.net.blobs['loss2up'].data 
            print '======================================================='
            #print " loss-sc3    : ", solver.net.blobs['loss3'].data
            print " loss-sc3-up : ", solver.net.blobs['loss3up'].data
            print '======================================================='
            print " loss-fuse   : ", solver.net.blobs['loss'].data 

            # store batch error
            batch_error.append(np.array(solver.net.blobs['loss'].data))

            if (it % test_interval == 0) and (it != 0):
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
                    solver.test_nets[0].params[ll][0].data[:, :, :, :] = solver.net.params[ll][0].data[:, :, :, :]
                    solver.test_nets[0].params[ll][1].data[:] = solver.net.params[ll][1].data[:]

                # initialize
                correct_sc1=0
                correct_sc2 = 0
                correct_sc3 = 0
                correct_fuse = 0
                error_class_contours = 0

                for test_it in range(test_iter):
                    solver.test_nets[0].forward()

                    # detect all correct predictions for this batch of images
                    temp_correct_sc1 = solver.test_nets[0].blobs['score1'].data.argmax(1) == solver.test_nets[0].blobs['label'].data[:, 0, :, :]
                    temp_correct_sc2 = solver.test_nets[0].blobs['score2up'].data.argmax(1) == solver.test_nets[0].blobs['label'].data[:, 0, :, :]
                    temp_correct_sc3 = solver.test_nets[0].blobs['score3up'].data.argmax(1) == solver.test_nets[0].blobs['label'].data[:, 0, :, :]
                    temp_correct_fuse = solver.test_nets[0].blobs['score'].data.argmax(1) == solver.test_nets[0].blobs['label'].data[:, 0, :, :]

                    # downsampled-labels-sc2
                    #temp_correct_sc2 = solver.test_nets[0].blobs['score2'].data.argmax(1) == solver.test_nets[0].blobs['label_sc2'].data[:, 0, :, :]
                    # downsampled-label-sc3
                    #temp_correct_sc3 = solver.test_nets[0].blobs['score3'].data.argmax(1) == solver.test_nets[0].blobs['label_sc3'].data[:, 0, :, :]

                    temp_error_class_contours = solver.test_nets[0].blobs['edge-loss'].data.item(0)

                    # store their sum
                    temp_correct_sc1 = np.float32(temp_correct_sc1.sum())
                    temp_correct_sc2 = np.float32(temp_correct_sc2.sum())
                    temp_correct_sc3 = np.float32(temp_correct_sc3.sum())
                    temp_correct_fuse = np.float32(temp_correct_fuse.sum())

                    # accumulate to total sum
                    correct_sc1 += temp_correct_sc1
                    correct_sc2 += temp_correct_sc2
                    correct_sc3 += temp_correct_sc3
                    correct_fuse += temp_correct_fuse
                    error_class_contours += temp_error_class_contours

                    # clear
                    temp_correct_sc1 = 0
                    temp_correct_sc2 = 0
                    temp_correct_sc3 = 0
                    temp_correct_fuse = 0
                    temp_error_class_contours = 0

                testset_correct_prc_sc1 = correct_sc1 / (test_iter * batch_size * num_pixels_prediction_instance)
                testset_correct_prc_sc2 = correct_sc2 / (test_iter * batch_size * num_pixels_prediction_instance)
                testset_correct_prc_sc3 = correct_sc3 / (test_iter * batch_size * num_pixels_prediction_instance)
                testset_correct_prc_fuse = correct_fuse / (test_iter * batch_size * num_pixels_prediction_instance)
                error_class_contours = error_class_contours / test_iter

                # ---- when considering downsampled labels
                #testset_correct_prc_sc2 = correct_sc2 / (test_iter * batch_size * (num_pixels_prediction_instance/4.0))
                #testset_correct_prc_sc3 = correct_sc3 / (test_iter * batch_size * (num_pixels_prediction_instance/16.0))

                # clear
                correct = 0

                print "Class-Boundary Error  : ", error_class_contours
                print 'Scale-1 Accuracy      : ', testset_correct_prc_sc1
                print 'Scale-2 Accuracy      : ', testset_correct_prc_sc2
                print 'Scale-3 Accuracy      : ', testset_correct_prc_sc3
                print 'Scale fuse Accuracy   : ', testset_correct_prc_fuse
                print "------------------------------------------------------------------------------------"
                print '####################################################################################'
                print '####################################################################################'

                # store validation set accuracy per test iteration
                test_acc[it // test_interval] = testset_correct_prc_fuse

                # clear
                testset_correct_prc = 0

# ============================================================ #


if __name__ == "__main__":

# --------------- INPUTS -------------- #

    # select gpu
    idx_gpu = 1

    # set pre-trained weights for model
    base_weights = 'path-to-model/train_iter_57000.caffemodel'

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
