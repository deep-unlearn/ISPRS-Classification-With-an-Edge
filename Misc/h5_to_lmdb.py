import caffe
import lmdb
from PIL import Image
import numpy as np
import h5py as h5

# for debbuging
#import pdb
#pdb.set_trace()


def make_img_lmdb(path_to_h5_txt):

    # open text file
    h5_txt_file = open(path_to_h5_txt)

    # construct image LMDB
    # IMAGE LMDB
    image_lmdb = lmdb.open('image-lmdb', map_size=int(1e12))
    with image_lmdb.begin(write=True) as in_txn:

        idx_image = 0  # initialize index saver
        # loop through different H5 files
        for in_idx, in_ in enumerate(h5_txt_file):
            h5_file = h5.File(in_.strip(), 'r')
            img_data = np.array(h5_file['data'])

            # loop though the H5 data and assign the the the lmdb
            for i in range(img_data.shape[0]):
                image = img_data[i, :]
                im_to_db = caffe.io.array_to_datum(image.astype('float'))
                in_txn.put('{:0>10d}'.format(idx_image), im_to_db.SerializeToString())
                idx_image += 1
                print "------- Processed: ", i, " from ", img_data.shape[0], " -----"
            print "------------------------------------------------------------------"
            print " "
            print "------ Processing IMAGE H5 ", in_, " ------"
            print " "
            print "------------------------------------------------------------------"

    image_lmdb.close()


def make_dem_lmdb(path_to_h5_txt):

    # open text file
    h5_txt_file = open(path_to_h5_txt)

    # DEM LMDB
    dem_lmdb = lmdb.open('dem-lmdb', map_size=int(1e12))
    with dem_lmdb.begin(write=True) as in_txn:

        idx_dem = 0  # initialize index saver
        # loop through different H5 files
        for in_idx, in_ in enumerate(h5_txt_file):
            h5_file = h5.File(in_.strip())
            dem_data = np.array(h5_file['dem'])

            # loop though the H5 data and assign the the the lmdb
            for i in range(dem_data.shape[0]):
                dem = dem_data[i, :]
                dem_to_db = caffe.io.array_to_datum(dem.astype('float'))
                in_txn.put('{:0>10d}'.format(idx_dem), dem_to_db.SerializeToString())
                idx_dem += 1
                print "------- Processed: ", i, " from ", dem_data.shape[0], " -----"

            print "------------------------------------------------------------------"
            print " "
            print "------ Processing DEM H5: ", in_, " ------"
            print " "
            print "------------------------------------------------------------------"

    dem_lmdb.close()


def make_label_lmdb(path_to_h5_txt):

    # open text file
    h5_txt_file = open(path_to_h5_txt)
    # LABEL LMDB
    label_lmdb = lmdb.open('label-lmdb', map_size=int(1e12))
    with label_lmdb.begin(write=True) as in_txn:

        idx_label = 0  # initialize index saver
        # loop through different H5 files
        for in_idx, in_ in enumerate(h5_txt_file):
            h5_file = h5.File(in_.strip())
            label_data = np.array(h5_file['label'])

            # loop though the H5 data and assign the the the lmdb
            for i in range(label_data.shape[0]):
                label = label_data[i, :]
                label_to_db = caffe.io.array_to_datum(label.astype('int'))
                in_txn.put('{:0>10d}'.format(idx_label), label_to_db.SerializeToString())
                idx_label += 1
                print "------- Processed: ", i, " from ", label_data.shape[0], " -----"

            print "------------------------------------------------------------------"
            print " "
            print "------ Processing Labels H5: ", in_, " ------"
            print " "
            print "------------------------------------------------------------------"

    label_lmdb.close()


def make_edge_label_lmdb(path_to_h5_txt):

    # open text file
    h5_txt_file = open(path_to_h5_txt)

    # Edge-LABEL LMDB
    edge_label_lmdb = lmdb.open('edge-label-lmdb', map_size=int(1e12))
    with edge_label_lmdb.begin(write=True) as in_txn:

        idx_label = 0  # initialize index saver
        # loop through different H5 files
        for in_idx, in_ in enumerate(h5_txt_file):
            h5_file = h5.File(in_.strip())
            edge_label_data = np.array(h5_file['label'])

            # loop though the H5 data and assign the the the lmdb
            for i in range(edge_label_data.shape[0]):
                edge_label = edge_label_data[i, :]
                edge_label_to_db = caffe.io.array_to_datum(edge_label.astype('float'))
                in_txn.put('{:0>10d}'.format(idx_label), edge_label_to_db.SerializeToString())
                idx_label += 1
                print "------- Processed: ", i, " from ", edge_label_data.shape[0], " -----"

            print "------------------------------------------------------------------"
            print " "
            print "------ Processing Labels H5: ", in_, " ------"
            print " "
            print "------------------------------------------------------------------"

    edge_label_lmdb.close()

# ================================================================================ #

# Converts raw LABELS into LMDB data

# ================================================================================ #

def make_image_db_from_image(image_file):
    image_db = lmdb.open('image-lmdb', map_size=int(1e12))
    with image_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(image_file):
            # load image:
            # - as np.uint8 {0, ..., 255}
            # - in BGR (switch from RGB)
            # - in Channel x Height x Width order (switch from H x W x C)
            im = np.array(Image.open(in_.strip()))  # or load whatever nd-array you need
            im = im[:, :, ::-1]
            im = im[:, :, 0:-1]  # take only three bands - from 4
            im = im.transpose((2, 0, 1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    image_db.close()

# ================================================================================ #

# Converts raw Image data into LMDB format

# ================================================================================ #

def make_label_db_from_image(label_file):
    label_db = lmdb.open('label-lmdb', map_size=int(1e12))
    with label_db.begin(write=True) as in_txn:
        for in_idx, in_ in enumerate(label_file):
            im = np.array(Image.open(in_.strip()))  # or load whatever nd-array you need
            im = im[:, :, 3:4] > 1  #take only single band and binarize it
            im = im.astype('uint8')
            im = im.transpose((2, 0, 1))
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx), im_dat.SerializeToString())
    label_db.close()


# ================================================================================ #

# Methods that convert as set of LMDB image data into TIF

# ================================================================================ #


def db_float_to_img(db_name, ext):
    env = lmdb.open(db_name, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            x = caffe.io.datum_to_array(datum)
            x = x.transpose((1, 2, 0))
            x = x[:, :, ::-1]
            if x.shape[2] == 1:
                x = np.concatenate((x, x, x), axis=2)
            img = Image.fromarray(x.astype(np.uint8))
            img.save(ext+str(key)+'.tif')


def db_int_to_img(db_name,ext):
    env = lmdb.open(db_name, readonly=True)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value)
            flat_x = np.fromstring(datum.data, dtype=np.float64)
            x = flat_x.reshape(datum.channels, datum.height, datum.width)
            x = x[:, :, ::-1]
            img = Image.fromarray(x.astype(np.uint8))
            img.save(ext+str(key)+'.tif')


if __name__ == '__main__':

    """
    Multiple methods for converting HDF5 data into LMDB format

        make_img_lmdb  @ construct LMDB for the image data HDF5
    
        make_dem_lmdb  @ construct LMDB for nDSM or DSM HDF5 (separately)
    
        make_edge_lmdb @ construct LMDB for class-contours HDF5

        make_label_lmdb @ construct LMDB for annotated labels HDF5

        TODO:

             * methods for constructing LMDBs directly from image data also included (no need for HDF5) - check them before using 
             
             * method for constructing images from LMDB --- check them before use
    
    """

    
    # ======================= INPUT ========================= #
    
    # path to HDF5 data
    path_to_h5_txt_data = "/mnt/data1/_Dimitris/00_Data/03_KITTI_Aerial/00_Code/data_creation/data_out/HDF5/train_data/train_data.txt"


    # ============================================ #

    # Enable ONE at a time to construct LMDBs 

    #make_img_lmdb(path_to_h5_txt=path_to_h5_txt_data)
    #make_dem_lmdb(path_to_h5_txt=path_to_h5_txt_data)
    #make_edge_label_lmdb(path_to_h5_txt=path_to_h5_txt_data)
    make_label_lmdb(path_to_h5_txt=path_to_h5_txt_data)
