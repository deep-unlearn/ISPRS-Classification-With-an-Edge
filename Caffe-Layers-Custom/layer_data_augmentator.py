import caffe
#import random
import numpy as np
import skimage
import skimage.transform

#import pdb
#pdb.set_trace()


class RealtimeDataAugmentation(caffe.Layer):
    

    """
    This method applies data augmentation to an input image by considering two types of augmentations

        1. Affine transform {including = scale, translation, rotation, shearing}
        2. Simple flip and/or mirroring

    Input

    image : Multiband Image or arbitary size  (HEIGHT, WIDTH, CHANNELS)

    All hyperparameters are defined in the "data_augmentation method" below



    TODO
    
        * Modify layer so that hyperparameters for the augmentation are not explicitly defined here
          but are passes directly into the prototxt

        * Currently this version only allows batch-size equal to 1. Larger batches mess-up the alligmed.
          Fix this so larger batches are also possible

    """

    def setup(self, bottom, top):

        assert bottom[0].data.shape[0] == 1, "Currently augmentation works with single-batch input"
        assert bottom[0].data.shape[0] == top[0].data.shape[0]

    def reshape(self, bottom, top):
        
        # Copy shape from bottom
        top[0].reshape(*bottom[0].data.shape)
    

    def forward(self, bottom, top):
        
            # # TRANSFORMATIONS # #

        def translation_transformation(img):
            center_shift = np.array((img.shape[0], img.shape[1])) / 2. - 0.5
            tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
            tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)
            return tform_center, tform_uncenter


        def build_augmentation_transform(img, zoom=1.0, rotation=0, shear=0, translation=(0, 0)):

            tform_center, tform_uncenter = translation_transformation(img)
            tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
            tform = tform_center + tform_augment + tform_uncenter  # shift to center, augment, shift back (for the rotation/shearing)
            return tform


        def random_perturbation_transform(img, zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
            # random shift [-4, 4] - shift no longer needs to be integer!
            shift_x = np.random.uniform(*translation_range)
            shift_y = np.random.uniform(*translation_range)
            translation = (shift_x, shift_y)

            # random rotation [0, 360]
            rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

            # random shear [0, 5]
            shear = np.random.uniform(*shear_range)

            # # flip
            if do_flip and (np.random.randint(2) > 0): # flip half of the time
                shear += 180
                rotation += 180
                # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
                # So after that we rotate it another 180 degrees to get just the flip.

            # random zoom [0.9, 1.1]
            # zoom = np.random.uniform(*zoom_range)
            log_zoom_range = [np.log(z) for z in zoom_range]
            zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
            # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

            return build_augmentation_transform(img, zoom, rotation, shear, translation)


        def random_flip_mirroring(img):

            # random generator for flip and/ or mirorring
            rand_val = np.random.randint(3)

            if rand_val == 0:
                # apply flip
                tr_img = np.rot90(img)

            if rand_val == 1:
                # apply mirroring
                tr_img = img[:, ::-1, :]

            if rand_val == 2:
                # apply both
                tr_img = np.rot90(img)
                tr_img = tr_img[:, ::-1, :]

            return tr_img


        def fast_warp(img, tf, mode='reflect', background_value=0.0):
            """
            This wrapper function is about five times faster than skimage.transform.warp, for our use case.
            """
            m = tf._matrix
            img_wf = np.empty((img.shape[0], img.shape[1], img.shape[2]), dtype='float32')
            for k in xrange(img.shape[2]):
                img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=(img.shape[0], img.shape[1]), mode=mode, cval=background_value)
            return img_wf

        # ============================================================================================ #    


        # =============  INPUTS ===========  #

        # when random value is larger that this threshold apply simple flip/ mirror operation
        flip_threshold = 0.40

        # define initial augmentation parameters
        augmentation_params = {
            'zoom_range': (1, 1.01),  # 0
            'rotation_range': (0, 15),  # 3 
            'shear_range': (0, 8),
            'translation_range': (-5, 5),
        }
            
        # ============= PROCESS ========= #
        
        for ii in range(bottom[0].data.shape[0]):
        
            # Copy all data
            input_im = bottom[0].data[ii, :]

            # roll-axis to build image (H x W x Chan)
            input_im = np.rollaxis(input_im, 0, 3)

            # randomly select augmentation mode => simple flip / affine transform
            augmentation_mode = np.random.random(1)

            if augmentation_mode > flip_threshold:
                out_im = random_flip_mirroring(input_im)

            if augmentation_mode <= flip_threshold:
                # compute random transformation
                tform_augment = random_perturbation_transform(img=input_im, **augmentation_params)

                # apply random transformation
                out_im = fast_warp(input_im, tform_augment).astype('float32')

            # convert to caffe tensor (Chan x H x W)
            out_im = np.rollaxis(out_im, 2, 0)

            # store to blob-output
            top[0].data[ii, :] = out_im[:]

    def backward(self, top, propagate_down, bottom):
        pass
