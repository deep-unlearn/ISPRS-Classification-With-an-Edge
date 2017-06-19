__author__ = 'dmarmanis'
from sklearn.feature_extraction.image import extract_patches_2d as patch_extractor
import matplotlib

matplotlib.use('Agg')  # to avoid problems in servers with no dispay variables
import matplotlib.pyplot as plt
from itertools import product
import h5py as h5
from PIL import Image
import numpy as np
import caffe
import shutil
import pdb
import os


def sequential_image_loader(path_to_folder, index_reader):
    """

    Read in images sequentially and compute their mean per band

    """

    # intialize
    list_of_images = []

    for file in sorted(os.listdir(path_to_folder)):
        if file.endswith('tif'):
            list_of_images.append(file)

    # add check if list of images is empty - try jpeg
    if list_of_images == []:
        for file in sorted(os.listdir(path_to_folder)):
            if file.endswith('jpg'):
                list_of_images.append(file)

    # check if list of images is empty
    assert list_of_images != []

    # find image to be returned in this call
    image_to_read = list_of_images[index_reader]

    # read in image
    current_image = Image.open(os.path.join(path_to_folder, image_to_read))
    img = np.array(current_image)
    # print image_to_read

    if img.ndim > 2:

        # read as BGR as in training data
        img = img[:, :, ::-1]

        # initialize for storing values
        img_mean = np.zeros((3, 1))

        # red band mean
        img_mean[0] = img[:, :, 0].mean()
        # green band mean
        img_mean[1] = img[:, :, 1].mean()
        # blue band mean
        img_mean[2] = img[:, :, 2].mean()
    else:
        img_mean = img.mean()

    return img, img_mean, list_of_images


def image_preprocessing(img_data,
                        img_mean,
                        img_value_scaler,
                        dsm_data,
                        dsm_mean,
                        ndsm_data,
                        ndsm_mean,
                        dsm_value_scaler):

    # Preproce IMAGE data
    # substract mean per band
    # img_data[:, 0, :, :] = img_data[:, 0, :, :] - img_mean[0]
    # img_data[:, 1, :, :] = img_data[:, 1, :, :] - img_mean[1]
    # img_data[:, 2, :, :] = img_data[:, 2, :, :] - img_mean[2]

    # re-scale
    img_data = img_data * img_value_scaler

    # Preprocess DEM data
    # dem_data = dem_data - dem_mean
    dsm_data = dsm_data * dsm_value_scaler
    ndsm_data = ndsm_data * dsm_value_scaler

    return img_data, dsm_data, ndsm_data


# code modification to reconstruct patches back by summing them


def reconstruct_from_patches_2d(patches,
                                image_size):
    """

    Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, SUMMING the overlapping
    regions.
    Read more in the :ref:`User Guide <image_feature_extraction>`.
    Parameters

    ----------

    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed

    Returns
    -------

    image : array, shape = image_size
        the reconstructed image

    """

    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i:i + p_h, j:j + p_w] += p
    return img


def image_padder(image,
                 dsm,
                 ndsm,
                 patch_size):
    """

    This function naively pads the image with half the size patch-size
    in this way it is assured all the pixels of the image to be visited at least ones
    Maybe smaller pad is enough but in this naive way an invetigation of the image dimensions is unnesessary

    :param image: image to be padeed
    :param dem:  dem data to be padded
    :param patch_size: path size to process the data
    :return: a padded image  (naively)

    """

    padding_size = 2 * patch_size

    padded_image = np.zeros((image.shape[0] + padding_size,
                             image.shape[1] + padding_size, image.shape[2]),
                            dtype='float32')
    padded_image[0: image.shape[0], 0:image.shape[1], :] = image

    padded_dsm = np.zeros((dsm.shape[0] + padding_size,
                           dsm.shape[1] + padding_size), dtype='float32')
    padded_dsm[0: dsm.shape[0], 0:dsm.shape[1]] = dsm

    padded_ndsm = np.zeros((ndsm.shape[0] + padding_size,
                            ndsm.shape[1] + padding_size), dtype='float32')
    padded_ndsm[0: ndsm.shape[0], 0:ndsm.shape[1]] = ndsm

    return padded_image, padded_dsm, padded_ndsm


def image_depadder(image, patch_size):
    # this function de-pads the image which is introduced by
    # method "image_padder" (look above method)

    deppader_size = 2 * patch_size

    image = image[:-deppader_size, :-deppader_size, :]

    return image


def model_loader(list_of_models,list_of_weights):
    """

    :param list_of_models:
    :param list_of_weights:
    :param model_indexer:
    :return: loaded caffe model

    """
    # set uo gpu parameters
    caffe.set_mode_gpu()
    caffe.set_device(0)

    # load model and initialize
    solver = caffe.SGDSolver(list_of_models)
    solver.net.copy_from(list_of_weights)

    return solver


def image_saver_semantics(model_scores,
                          save_folder,
                          image_vis=None,
                          dem_vis=None,
                          vmin=None,
                          vmax=None):
    """

        save images and scores for Semantic_Network

    """

    # create folder for saving data
    os.makedirs(save_folder)

    # block pop-up figure
    plt.ioff()

    if image_vis is not None:
        fig = plt.figure()
        imgplt = plt.imshow(np.floor(image_vis +
                                     np.abs(image_vis.min())).astype('uint8'))
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, "Image"), dpi=400)

    if dem_vis is not None:
        imgplot = plt.imshow(dem_vis)
        plt.axis('off')
        plt.savefig(os.path.join(save_folder, "DEM"), dpi=400)

    out = model_scores.argmax(axis=2)
    imgplot = plt.imshow(out, vmin=0, vmax=6, cmap='gist_ncar')
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, "Classification"), dpi=400)

    plt.imshow(model_scores[:, :, 0])
    plt.title('Score-Impervious')
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, "Scores-Impervious-Surfaces"), dpi=400)

    plt.imshow(model_scores[:, :, 1])
    plt.title('Score-Buildings')
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, "Scores-Buildings"), dpi=400)

    plt.imshow(model_scores[:, :, 2])
    plt.title('Score-Vegetation')
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, "Scores-Low-Vegeration"), dpi=400)

    plt.imshow(model_scores[:, :, 3])
    plt.title('Score-Trees')
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, "Score-Trees"), dpi=400)

    plt.imshow(model_scores[:, :, 4])
    plt.title('Score-Cars')
    plt.axis('off')
    plt.savefig(os.path.join(save_folder, "Scores-Cars"), dpi=400)

    plt.close('all')


def hdf5_score_saver(score_matrix,
                     save_data_path,
                     naming_string_file,
                     attribute_name):
    """
    This function accumulates the scores and save them according to an interval defined by the
    "save_interval" variable

    :param net:
    :param save_interval:
    :param i_scoring:
    :param i_idx:
    :return:

    """

    # construct folder for saving outcome
    os.makedirs(save_data_path)

    # store score for last entry of matrix
    score_matrix = score_matrix.astype('float32')

    # string for saving name
    current_save_name = naming_string_file + '_h5_data'

    with h5.File(os.path.join(save_data_path, current_save_name) + '.h5', 'w') as ff:
        ff[attribute_name] = score_matrix

    # save txt file with h5 names
    with open(os.path.join(save_data_path, current_save_name) + '_h5_data' + '.txt', 'w') as f:
        f.write(os.path.join(save_data_path, current_save_name) + '.h5\n')


def model_inference(solver,
                    image_for_inference,
                    dsm_for_inference,
                    ndsm_for_inference,
                    size_of_patches,
                    stride_step,
                    image_mean,
                    dsm_mean,
                    ndsm_mean,
                    img_scaler,
                    dsm_scaler):
    if dsm_for_inference is not None:

        # prediction step - for faster inference, will define over how many pixels an inference will be perfomed
        # in x and y direction !!!! This is defined by the STRIDE_STEP variable in method input

        # image padding to ensure that image will be covered completely through the adaptive prodiction
        # overllaping step !!!!
        image_for_inference, \
        dsm_for_inference, \
        ndsm_for_inference = image_padder(image_for_inference,
                                          dsm_for_inference,
                                          ndsm_for_inference,
                                          size_of_patches)

        # image_shape
        img_height, img_width = image_for_inference.shape[0], image_for_inference.shape[1]

        # this model scores - not the global cummulative scores of all models
        this_model_scores = np.zeros((img_height, img_width, 5))

        # detect point for starting and stopping the inference
        # - so that patch size will cover the complete image
        start_point = (size_of_patches / 2) + 1
        stop_point_x = img_height - (size_of_patches / 2)
        # stop_point_y = img_width - (size_of_patches/2)

        # extract patch-size row of data for inference
        # for i_row in range(start_point, stop_point_x + 1):
        for i_row in range(start_point, stop_point_x + 1, stride_step):

            row_image = image_for_inference[i_row - (size_of_patches / 2) - 1: i_row + (size_of_patches / 2), :]
            row_dsm = dsm_for_inference[i_row - (size_of_patches / 2) - 1: i_row + (size_of_patches / 2), :]
            row_ndsm = ndsm_for_inference[i_row - (size_of_patches / 2) - 1: i_row + (size_of_patches / 2), :]

            # extract dense grid patches
            row_image_patches = patch_extractor(image=row_image,
                                                patch_size=(size_of_patches, size_of_patches),
                                                max_patches=None)
            row_dsm_patches = patch_extractor(image=row_dsm,
                                              patch_size=(size_of_patches, size_of_patches),
                                              max_patches=None)
            row_ndsm_patches = patch_extractor(image=row_ndsm,
                                               patch_size=(size_of_patches, size_of_patches),
                                               max_patches=None)

            # reshape to CAFFE standards
            row_image_patches = np.rollaxis(row_image_patches, 3, 1)
            row_dsm_patches = np.rollaxis(row_dsm_patches[:, None, :, :], 1, 1)
            row_ndsm_patches = np.rollaxis(row_ndsm_patches[:, None, :, :], 1, 1)

            if row_image_patches is None and row_dsm_patches is None and row_ndsm_patches is None:
                assert False, "Please check data cause trying to make prediction with no Image-data and/or no DEM-data"

            assert row_image_patches.shape[0] == row_dsm_patches.shape[0] \
                                              == row_ndsm_patches.shape[0], \
                "Number of DEM and IMAGE patches are not the " \
                "same, please check why and ensure that they " \
                "are equal"

            # convert  patches to float-32
            row_image_patches = np.array(row_image_patches, dtype='float32')
            row_dsm_patches = np.array(row_dsm_patches, dtype='float32')
            row_ndsm_patches = np.array(row_ndsm_patches, dtype='float32')

            # subtract mean and normalize data - as in TRAINING protxt
            row_image_patches, \
            row_dsm_patches, \
            row_ndsm_patches = image_preprocessing(img_data=row_image_patches,
                                                   img_mean=image_mean,
                                                   img_value_scaler=img_scaler,
                                                   dsm_data=row_dsm_patches,
                                                   dsm_mean=dsm_mean,
                                                   ndsm_data=row_ndsm_patches,
                                                   ndsm_mean=ndsm_mean,
                                                   dsm_value_scaler=dsm_scaler)


            # TODO  ---> include self adaptive number of batches for the caffe model - currently the batches is standardize to size 1

            # initialize score patch saver --- FOR 5 CLASSES
            score_patches = np.zeros((row_dsm_patches.shape[0], 5, size_of_patches, size_of_patches))

            # TODO - REPLACE BACK FOR COMPLETE LOOPS

            for i_batch in range(0, row_image_patches.shape[0], stride_step):
                # use caffe model for inference
                solver.net.blobs['image'].data[...] = row_image_patches[i_batch, :]
                solver.net.blobs['dsm'].data[...] = row_dsm_patches[i_batch, :]
                solver.net.blobs['ndsm'].data[...] = row_ndsm_patches[i_batch, :]

                # apply feedforward pass
                solver.net.forward()

                # print "Processing column :", i_batch,  "  from ", row_image_patches.shape[0]

                # store scoring
                # score_patches[i_batch, :, :, :] = solver.net.blobs['score'].data[:, :5, :, :]
                score_patches[i_batch, :, :, :] = solver.net.blobs['prob'].data[:, :5, :, :]

                # TO DELETE score_patches = score_patches[:,:3,:,:]
                # TO DELETE score_patches[i_batch] = np.rollaxis(row_image_patches, 2,3)[i_batch]

            # reshape to scikit learn standards (b,0,1,c)
            score_patches = np.rollaxis(score_patches, 1, 4)

            # reconstruct by overlapping summations
            row_scores = reconstruct_from_patches_2d(score_patches,
                                                     (row_image.shape[0], row_image.shape[1], 5))

            # store to score matrix
            this_model_scores[i_row - (size_of_patches / 2) - 1: i_row + (size_of_patches / 2), :, :] = row_scores

            print "----------------------------------------------------------------------------------------------------"
            print "----------------------------------------------------------------------------------------------------"
            print "---------------------- Done with row : ", i_row, " from ", stop_point_x, " -------------------------"
            print "----------------------------------------------------------------------------------------------------"
            print "----------------------------------------------------------------------------------------------------"

        # remove padding introduced before to ensure complete image inference
        this_model_scores = image_depadder(this_model_scores, size_of_patches)

        return this_model_scores


def cumulative_model_inference(path_to_folder_with_images,
                               path_to_folder_with_dsm,
                               path_to_folder_with_ndsm,
                               patch_size,
                               model_list,
                               weight_list,
                               img_save_data_path,
                               hdf5_save_data_path,
                               string_names,
                               hdf5_attributes,
                               stride_value,
                               img_scaler,
                               dsm_scaler,
                               save_visualizations=False):
    """
    THis method applied sequential inference and recostruction of data from patches to a set of models
    and summs up their individual score maps

    """

    # initialization
    num_of_images = 0

    # this index will keep track of the current image to be loaded
    img_index_reader = 0

    # index for saving folder
    save_data_index = 0

    # ------ computation calculation ----- #
    # - number of  images to process - #
    for file in os.listdir(path_to_folder_with_images):
        if file.endswith('tif'):
            num_of_images += 1

    # ========================== SEQUENTIAL PROCESSING ============================ #

    # loop number of images number of
    for i in range(num_of_images):

        # call image
        complete_image, image_mean, img_list = \
            sequential_image_loader(path_to_folder_with_images, img_index_reader)
        complete_dsm, dsm_mean, dsm_list = \
            sequential_image_loader(path_to_folder_with_dsm, img_index_reader)
        complete_ndsm, ndsm_mean, dem_list = \
            sequential_image_loader(path_to_folder_with_ndsm, img_index_reader)

        assert complete_image.shape[0] == complete_dsm.shape[0] and \
               complete_image.shape[1] == complete_dsm.shape[1] and \
               complete_image.shape[0] == complete_ndsm.shape[0] and \
               complete_image.shape[1] == complete_ndsm.shape[1], \
            "Shape of Image and DEM does not match, please ensure " \
            "that are correctly read-in with associative order"

        # increase index reader
        img_index_reader += 1

        # initialize image for storing scores
        cumulative_img_scores = np.zeros((complete_image.shape[0],
                                          complete_image.shape[1], 5),
                                         dtype='float32')

        # loop through each model and compute inference
        for i_model in range(len(model_list)):
            solver_model = model_loader(list_of_models=model_list,
                                        list_of_weights=weight_list)

            # loop through the various models and perform inference
            image_scores = model_inference(solver=solver_model,
                                           image_for_inference=complete_image,
                                           dsm_for_inference=complete_dsm,
                                           ndsm_for_inference=complete_ndsm,
                                           size_of_patches=patch_size,
                                           stride_step=stride_value,
                                           image_mean=image_mean,
                                           dsm_mean=dsm_mean,
                                           ndsm_mean=ndsm_mean,
                                           img_scaler=img_scaler,
                                           dsm_scaler=dsm_scaler)

            save_name_string = img_list[save_data_index]

            if save_visualizations is True:
                # save image outcomes
                image_saver_semantics(model_scores=image_scores,
                                      save_folder=os.path.join(img_save_data_path[i_model],
                                                               str(save_name_string)),
                                      image_vis=complete_image,
                                      dem_vis=complete_dsm,
                                      vmin=0,
                                      vmax=6)

            # save scores in hdf5 format matrix
            hdf5_score_saver(score_matrix=image_scores,
                             save_data_path=os.path.join(hdf5_save_data_path[i_model],
                                                         str(save_name_string)),
                             naming_string_file=string_names[i_model],
                             attribute_name=hdf5_attributes[i_model])

            # accumulate all models scores
            cumulative_img_scores += image_scores

        if save_visualizations is True:
            # save cumulative image scores
            image_saver_semantics(model_scores=cumulative_img_scores,
                                  save_folder=os.path.join(img_save_data_path[i_model + 1],
                                                           str(save_name_string)),
                                  image_vis=complete_image,
                                  dem_vis=complete_dsm,
                                  vmin=0,
                                  vmax=6)

        # save cummulative scores in hdf5 matrix
        hdf5_score_saver(score_matrix=cumulative_img_scores,
                         save_data_path=os.path.join(hdf5_save_data_path[i_model + 1],
                                                     str(save_name_string)),
                         naming_string_file=string_names[i_model + 1],
                         attribute_name=hdf5_attributes[i_model + 1])

        # increase save folder index
        save_data_index += 1

        print "DONE"



# =========================================== MAIN =========================================== #

if __name__ == '__main__':
    # set for debug
    pdb.set_trace()

    # ======================== SET INFERENCE INPUTS =============== #

    patch_size = 256
    stride = 220

    # take this values from TRAINING prototxt
    dsm_value_scaler = 0.003333333
    img_value_scaler = 0.00390625

    # This will create graphics of all inferred data
    generate_visualizations = True

    path_to_folder_with_images = '...path to folder with images'
    path_to_folder_with_DSMs = '...path to folder with DSM'
    path_to_folder_with_nDSMs = "...path to folder with nDSM"

    # inference models
    #  -------- MODEL NET PARAMETERS -------- #
    model_solver_path = '...path to folder with solver'
    model_solver = '...name of solver'

    model_weights_path = '...path to folder with weights'
    model_weights = '... weights name caffemodel'

    model_save_string_name = 'segnet_scores'

    model_attribute_hdf5_name = 'segnet_scores'

    # create storing folders if do not exist
    annot_image_dir = './annotated_image' + str(stride) + '_pix_overlap'
    h5_save_data_dir = './h5_data' + str(stride) + '_pix_overlap'

    if not os.path.exists(annot_image_dir):
        os.makedirs(annot_image_dir)

    if not os.path.exists(h5_save_data_dir):
        os.makedirs(h5_save_data_dir)

    # apply sequential inference in a set of multiple or single models and store data
    cumulative_model_inference(path_to_folder_with_images=path_to_folder_with_images,
                               path_to_folder_with_dsm=path_to_folder_with_DSMs,
                               path_to_folder_with_ndsm=path_to_folder_with_nDSMs,
                               img_scaler=img_value_scaler,
                               dsm_scaler=dsm_value_scaler,
                               patch_size=patch_size,
                               stride_value=stride,
                               model_list=os.path.join(model_solver_path, model_solver),
                               weight_list=os.path.join(model_weights_path, model_weights),
                               img_save_data_path=annot_image_dir,
                               hdf5_save_data_path=h5_save_data_dir,
                               string_names=model_save_string_name,
                               hdf5_attributes=model_attribute_hdf5_name,
                               save_visualizations=generate_visualizations)
