__author__ = 'deep_unlearn'

from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image
import numpy as np
import os
from natsort import natsorted
from osgeo import gdal
import h5py 


"""
   Code for extracting randomly patches from a set of image scenes  - 

      - If exist considers patches from a DSM & nDSM
      - If height information(nDSM, DEM)  is not available extracts data from RGB 
           images and respective labels only
   
    Results are stored in limited-size H5 format for use with CAFFE-lib

    Optional: The code also accepts a mask with where the allowed foreground for 
              patch extraction is explicitly designated. This also allows partial 
              overal with background.

"""

# ======================================== #

# Enable for degub mode

#import pdb
#pdb.set_trace()
#from matplotlib import pyplot as plt

# ######################################


def remove_background(img_patches, 
                      dsm_patches, 
                      ndsm_patches, 
                      label_patches, 
                      mask_patches):

    # initialize storing index
    idx_del = []

    for i in range (mask_patches.shape[0]):
        total_sum_mask = 0
        current_patch = mask_patches[i, :]
        total_sum_mask = current_patch.sum()  # gives the total value of mask pixel-values - 255 for non background pixel

        # if non-black pixel as less than 20% of total size remove
        if total_sum_mask < 3342336: 
            
            # strore index
            idx_del.append(i)

    # delete found indeces
    img_patches = np.delete(img_patches, idx_del, axis=0)
    label_patches = np.delete(label_patches, idx_del, axis=0)

    if dsm_patches is not None:
        dsm_patches = np.delete(dsm_patches, idx_del, axis=0)

    if ndsm_patches is not None:
        ndsm_patches = np.delete(ndsm_patches, idx_del, axis=0)

    if mask_patches is not None:
        mask_patches = np.delete(mask_patches, idx_del, axis=0)

    assert (img_patches.shape[0]==label_patches.shape[0])

    return img_patches, dsm_patches, ndsm_patches, label_patches
    

def read_in_images(img_path, dsm_path, ndsm_path, labels_path, mask_path):

    # read in and convert to array
    image = Image.open(img_path)
    img = np.array(image)

    if dsm_path is not None:
        file = gdal.Open(dsm_path)
        dsm = np.array(file.GetRasterBand(1).ReadAsArray())
    
        # check that dem does not contain nans or infinite - 
        # if exist replace spikes with average height
        if ~(np.isfinite(dsm.sum()) or np.isfinite(dsm).all()):
            idx_nan = np.isnan(dsm)
            mean_val = dsm[~np.isnan(dsm)].mean()
            dsm[idx_nan] = mean_val
    else:
        dsm = None

    if ndsm_path is not None:    
        file = gdal.Open(ndsm_path)
        ndsm = np.array(file.GetRasterBand(1).ReadAsArray())

        # check that ndsm does not contain nans or infinite
        if ~(np.isfinite(ndsm.sum()) or np.isfinite(ndsm).all()):
            idx_nan = np.isnan(ndsm)
            mean_val = ndsm[~np.isnan(ndsm)].mean()
            ndsm[idx_nan] = mean_val
    else:
        ndsm = None

    labels = Image.open(labels_path)
    labels = np.array(labels)
    
    if mask_path is not None:
        file = gdal.Open(mask_path)
        mask = np.array(file.GetRasterBand(1).ReadAsArray())
    else:
        mask = None

    return img, dsm, ndsm, labels, mask


def patch_extractor(img=None, 
                    dsm=None, 
                    ndsm=None, 
                    labels=None, 
                    mask_ext=None, 
                    num_patches=None, 
                    patch_size=None, 
                    random_num='None'):

    # extract image patches
    print "Random number - patch extraction : ", random_num

    img_patches = extract_patches_2d(img, 
                                     (patch_size, patch_size), 
                                     num_patches, 
                                     random_state=random_num)
    
    # if DSM exist extract patches
    if dsm is not None:
        dsm_patches = extract_patches_2d(dsm, 
                                         (patch_size, patch_size), 
                                         num_patches, 
                                         random_state=random_num)
    else: 
        dsm_patches = None

    # if nDSM exist extract patches
    if ndsm is not None:
        ndsm_patches = extract_patches_2d(ndsm, 
                                          (patch_size, patch_size), 
                                          num_patches, 
                                          random_state=random_num)
    else:
        ndsm_patches = None

    # if nDSM exist extract patches
    if mask_ext is not None:
        mask_patches = extract_patches_2d(mask_ext, 
                                          (patch_size, patch_size), 
                                          num_patches, 
                                          random_state=random_num)
        mask_patches=mask_patches[:, :, :, None]
    else:
        mask_patches = None
        
    # extract label patches
    label_patches = extract_patches_2d(labels, 
                                       (patch_size, patch_size), 
                                       num_patches, 
                                       random_state=random_num)


    # =========== Remove Tiles containing  only background ============ #

    img_patches, dsm_patches, ndsm_patches, label_patches = remove_background(img_patches, 
                                                                              dsm_patches, 
                                                                              ndsm_patches,
                                                                              label_patches, 
                                                                              mask_patches)

    return img_patches, dsm_patches, ndsm_patches, label_patches


def crop_labels(img_labels=None, keep_area=None, patch_size=None):

    """

    This method allow to extract sub-patches from patches
    Used for smaller context in label

    """

    #  index of center pixel to be extracted
    middle_element = np.floor(patch_size/2)
    pixels_around_center = np.floor(keep_area/2)

    # initialize matrix to store new sub-labels
    img_labels_new = np.zeros((img_labels.shape[0], 
                               keep_area*keep_area*img_labels.shape[3]))

    # Crop labels so that a small sub-window of area "keep_area" is retained after process
    for i in xrange(img_labels.shape[0]):

        # read patch sequentially and store it to temporary value
        current_temp = img_labels[i, :, :, :]

        # new set of minimized labels
        label_temp = current_temp[middle_element - \
                                  pixels_around_center: middle_element + \
                                  pixels_around_center + 1,
                                  middle_element - pixels_around_center: middle_element \
                                  + pixels_around_center + 1]

        label_temp = label_temp.flatten()
        # store into new matrix
        img_labels_new[i, :] = label_temp

    return img_labels_new


def label_replacer(img_labels=None):

    """
    Method replaces standard RGB label-values with single integer
    categorical variable. Define initial label below

    """

    # Building     (255, 0, 0)       --- 0
    # Road         (255, 105, 180)   --- 1
    # Sidewalk     (0, 0, 255)       --- 2
    # Parking      (255, 255, 0)     --- 3
    # Background   (0,0, 0)          --- 4

    # indeces to detect
    building = (255, 0, 0)
    road = (255, 105, 180)
    sidewalk = (0, 0, 255)
    parking = (255, 255, 0)
    background = (0, 0, 0)

    # initialize matrix to store new labels
    img_labels_single_id = np.zeros((img_labels.shape[0], 
                                     img_labels.shape[1], 
                                     img_labels.shape[2], 1))

    for i in range(img_labels.shape[0]):

        # store temp image to process
        temp_img = img_labels[i, :, :, :]

        xx_idx_building, yy_idx_building = np.where(np.all(temp_img == building, axis=-1))
        xx_idx_road, yy_idx_road = np.where(np.all(temp_img == road, axis=-1))
        xx_idx_sidewalk, yy_idx_sidewalk = np.where(np.all(temp_img == sidewalk, axis=-1))
        xx_idx_parking, yy_idx_parking = np.where(np.all(temp_img == parking, axis=-1))
        xx_idx_background, yy_idx_background = np.where(np.all(temp_img == background, axis=-1))

        # replace with new single indexes
        temp_img[xx_idx_building, yy_idx_building] = 0
        temp_img[xx_idx_road, yy_idx_road] = 1
        temp_img[xx_idx_sidewalk, yy_idx_sidewalk] = 2
        temp_img[xx_idx_parking, yy_idx_parking] = 3
        temp_img[xx_idx_background, yy_idx_background] = 4  

        img_labels_single_id[i, :, :, :] = temp_img[:, :, 0:1]

    return img_labels_single_id


def store_images(img_patches, 
                 img_labels, 
                 path_store_img, 
                 path_store_label):

    """
        Method for storing IMAGE and LABEL patches in folder - used with MemoryDataLayer 
        in CAFFE


    Todo:
           * Expand this for considering nDSM, DEM patches
           
    """
    
    for ii in range(img_labels.shape[0]):

        temp_img = img_patches[ii, :, :, :]
        temp_label = img_labels[ii, :, :, 0]

        img = Image.fromarray(temp_img)
        label = Image.fromarray(temp_label)

        name_image = 'image_' + str(ii) + '.tif'
        name_label = 'label_' + str(ii) + '.tif'

        img.save(os.path.join(path_store_img, name_image), "TIFF")
        label.save(os.path.join(path_store_label, name_label), "TIFF")


def construct_hdf5_dataset(img_patches, 
                           dsm_patches, 
                           ndsm_patches, 
                           img_labels, 
                           save_path, 
                           save_file_name, 
                           band_mean_vals=None):

    img_patches = img_patches.astype('float32')
    img_labels = img_labels.astype('float32')
    if dsm_patches is not None:
        dsm_patches = dsm_patches.astype('float32')
    if ndsm_patches is not None:
        ndsm_patches = dsm_patches.astype('float32')
    
    if band_mean_vals is not None:
        img_patches[:, 0, :, :] = img_patches[:, 0, :, :] - band_mean_vals[0]  # R
        img_patches[:, 1, :, :] = img_patches[:, 1, :, :] - band_mean_vals[1]  # G
        img_patches[:, 2, :, :] = img_patches[:, 2, :, :] - band_mean_vals[2]  # B

    # - transpose channels from RGB to BGR
    img_patches = img_patches[:, :, :, ::-1]

    # - convert to Batches x Channel x Height x Width order (switch from B x H x W x C)
    img_patches = img_patches.transpose((0, 3, 1, 2))
    if dsm_patches is not None:
        dsm_patches = dsm_patches.transpose((0, 3, 1, 2))
    if ndsm_patches is not None:
        ndsm_patches = ndsm_patches.transpose((0, 3, 1, 2))
    img_labels = img_labels.transpose((0, 3, 1, 2))

    # compute total size
    total_size = img_patches[0] * img_patches[1] * img_patches[2] * img_patches[3]

    # STORE DATA AS UNCOMPRESSED HDF5
    with h5py.File(save_path + '/' + save_file_name + '.h5', 'w') as f:
        f['data'] = img_patches
        if dsm_patches is not None:
            f['dsm'] = dsm_patches
        if ndsm_patches is not None:
            f['ndsm'] = ndsm_patches
        f['label'] = img_labels

    with open(save_path + '/' + save_file_name + '.txt', 'w') as f:
        f.write(save_path + '/' +save_file_name + '.h5\n')


# ================== MAIN ===================== #


def main (image_path_folder, 
              labels_path_folder,
              allow_perturbation, 
              num_patches, 
              patch_size, 
              save_name, 
              save_path, 
              random_num,
              dsm_path_folder=None, 
              ndsm_path_folder=None,
              mask_path_folder=None):
    
    # store names of images to be processed
    image_name_list = natsorted(os.listdir(image_path_folder))
    labels_name_list = natsorted(os.listdir(labels_path_folder))
    
    # sort naturally if exist
    if dsm_path_folder is not None:
        dsm_name_list = natsorted(os.listdir(dsm_path_folder))
    else:
        dsm_name_list = None

    if ndsm_path_folder is not None:
        ndsm_name_list = natsorted(os.listdir(ndsm_path_folder))
    else:
        ndsm_name_list = None
    
    if mask_path_folder is not None:
        mask_name_list = natsorted(os.listdir(mask_path_folder))
    else:
        mask_name_list = None

    # CHECK

    # ensure that num of labels = num of images
    
    if (dsm_name_list is not None) and (ndsm_name_list is not None):
        assert (len(labels_name_list) == len(image_name_list) == len(dsm_name_list) == len(ndsm_name_list))

    else: # only images-labels-mask exist
        assert (len(labels_name_list) == len(image_name_list) == len(mask_name_list))
    
    # initialize
    final_img_patches = []
    final_dsm_patches = []
    final_ndsm_patches = []
    final_labels = []

    #index
    i = 0

    # read in tiff images from folder
    for file_name in natsorted(os.listdir(image_path_folder)):

        # temp store image_name and label_name
        image_path = os.path.join(image_path_folder, image_name_list[i])
        
        labels_path = os.path.join(labels_path_folder, labels_name_list[i])
        
        if dsm_path_folder is not None:
            dsm_path = os.path.join(dsm_path_folder, dsm_name_list[i])
        else:
            dsm_path = None
            
        if ndsm_path_folder is not None:
            ndsm_path = os.path.join(ndsm_path_folder, ndsm_name_list[i])
        else:
            ndsm_path = None
        
        if mask_path_folder is not None:
            mask_path = os.path.join(mask_path_folder, mask_name_list[i])
        else:
            mask_path = None

        # read in image and labels
        img, dsm, ndsm, labels, mask = read_in_images(image_path, dsm_path, ndsm_path, labels_path, mask_path)

        # extract number of patches randomly
        img_patches, \
        dsm_patches, \
        ndsm_patches, \
        img_labels = patch_extractor(img=img,
                                     dsm=dsm,
                                     ndsm=ndsm,
                                     labels=labels,
                                     mask_ext=mask,
                                     num_patches=num_patches,
                                     patch_size=patch_size,
                                     random_num=random_num)

        img_patches = np.array(img_patches, dtype='float32')
        
        if dsm_patches is not None:
            dsm_patches = np.array(dsm_patches, dtype='float32')
        
        if ndsm_patches is not None:
            ndsm_patches = np.array(ndsm_patches, dtype='float32')            

        # replace label values for categorical classes # 
        # If label is single-plane edge then skip this
        if img_labels.ndim > 3:
            # replace standard RGB labels with single value labels
            img_labels = label_replacer(img_labels)

        # extract sub-patch of labels - smaller than initial size 
        # TODO: NEEDS MODIFICATION for including dem-data
        # img_labels = crop_labels(img_labels, 
        #                          keep_area=keep_area, 
        #                          patch_size=patch_size)

        # store cumulative data
        if i == 0:
            final_img_patches = img_patches
            final_dsm_patches = dsm_patches
            final_ndsm_patches = ndsm_patches
            final_labels = img_labels
        else:
            final_img_patches = np.concatenate((final_img_patches, 
                                                img_patches), 
                                               axis=0)
            if final_dsm_patches is not None:
                final_dsm_patches = np.concatenate((final_dsm_patches, 
                                                    dsm_patches), 
                                                   axis=0)
            if final_ndsm_patches is not None:
                final_ndsm_patches = np.concatenate((final_ndsm_patches, 
                                                     ndsm_patches), 
                                                    axis=0)
            final_labels = np.concatenate((final_labels, img_labels), 
                                          axis=0)

        # increase index
        i += 1

    # add singleton dimension in dem_patches so that makes up a 4D -array
    # as required in Caffe
    if final_dsm_patches is not None:
        final_dsm_patches = final_dsm_patches[:, :, :, np.newaxis]

    if final_ndsm_patches is not None:
        final_ndsm_patches = final_ndsm_patches[:, :, :, np.newaxis]

    #if labels are edges also add singleton dimension to them
    if final_labels.ndim < 4:    
        final_labels = final_labels[:, :, :, np.newaxis]

    # Perturb training instances and labels equally
    # perturbation index
    if allow_perturbation is True:
        perturbation_index = np.random.permutation(final_img_patches.shape[0])

        final_img_patches = final_img_patches[perturbation_index]

        if final_dsm_patches is not None:
            final_dsm_patches = final_dsm_patches[perturbation_index]
        if final_ndsm_patches is not None:
            final_ndsm_patches = final_ndsm_patches[perturbation_index]

        final_labels = final_labels[perturbation_index]

    # ensure labels are integers - then float32
    final_labels = np.array(final_labels, dtype='int32')
    final_labels = np.array(final_labels, dtype='float32')

    # if saving variables exist save HDf5 file
    if 'save_name' and 'save_path' in locals():

        #Construct HDF5 data
        construct_hdf5_dataset(img_patches=final_img_patches, 
                               dsm_patches=final_dsm_patches, 
                               ndsm_patches=final_ndsm_patches, 
                               img_labels=final_labels,
                               save_file_name=save_name, 
                               save_path=save_path, band_mean_vals=None)

    # Store images and labels
    # store_images(img_patches=final_patches, img_labels=final_labels,
    #             path_store_img=save_path_img, path_store_label=save_path_label)


#############################################################################################################


if __name__ == '__main__':


    # ===============  INPUTs ================ #

    # Path to images
    path_img_fold = "/mnt/data1/_Dimitris/00_RS_Data/03_KITTI_Aerial/Aerial/test_images/images"
    
    # Path to DSM
    path_dsm_fold = None
    
    # Path to nDSM
    path_ndsm_fold = None
    
    # Path to image masks
    path_mask_fold = "/mnt/data1/_Dimitris/00_RS_Data/03_KITTI_Aerial/Aerial/test_images/extent_masks/"
    
    # Labels 
    path_label_folder = "/mnt/data1/_Dimitris/00_RS_Data/03_KITTI_Aerial/Aerial/test_images/labels"

    # Boolean - Enable if to randomly perurb data
    perturb_mode = True

    """
    IMPORTANT
    
         TOTAL NUMBER of patches : 
    
             (NUM of patches PER image (variable below)) x (length of random number list - defined below) 

    """

    num_of_patches_per_img = 200  # number of patches PER each input image !!!!!!!!
    
    # size in pixels of patches
    size_of_patch = 256

    # ONLY enable if want to create HDF5 data - otherwise comment out
    saving_path = "./test/"

    random_numbers = [83762,  
                      38476,   
                      26152,   
                      38485,    
                      2221,
                      213875, 
                      968564,  
                      3735251, 
                      78705038, 
                      5342]

    # ONLY enable if want to create HDF5 data - otherwise comment out
    saving_names = ["train_data_256x256_1", 
                    "train_data_256x256_2", 
                    "train_data_256x256_3",
                    "train_data_256x256_4", 
                    "train_data_256x256_5", 
                    "train_data_256x256_6",
                    "train_data_256x256_7", 
                    "train_data_256x256_8", 
                    "train_data_256x256_9",
                    "train_data_256x256_10"]

    # ============================================================ #

    for i in range(len(random_numbers)):

        print "\t\t ", i+1, " out of ", len(random_numbers)

        # read in next random number
        random_num = random_numbers[i]

        saving_name = saving_names[i]
        main(image_path_folder=path_img_fold, 
                 dsm_path_folder=path_dsm_fold,
                 ndsm_path_folder=path_dsm_fold,
                 labels_path_folder=path_label_folder, 
                 mask_path_folder = path_mask_fold,
                 allow_perturbation=perturb_mode,
                 num_patches=num_of_patches_per_img, 
                 patch_size=size_of_patch, 
                 save_name=saving_name,
                 save_path=saving_path, 
                 random_num=random_num)

    print "DONE"
