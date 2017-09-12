# Class-Boundaries-Implementation


 ======= INFO =======
 
 This repository contains the code for replicating results in the following work
 
 "Classification with An Edge: Improving Semantic Image Segmentation with Boundary Detection" 
 
    https://arxiv.org/abs/1612.01337
 
 The repository contains code:
 
    - Training an ensemble of three independent fully convolutional networks (VGG16-FCN, Pascal-FCN, SegNet-FCN)
      with respective pre-trained models over the ISPRS-Vaihingen & ISPRS-Potsdam dataset - Uisng Caffe Deep-Learning Framework
    
    - Some miscelenous python files for data construction (Patch-extraction, HDF5 & LMDB generation)

    - Some sample code on training models and respective prototxt files

    - A Caffe custom Python-Layer for stochastic data-augmentation of data on every training batch
 
 
 If you make use of the code please cite the following work:

  @article{marmanis2016classification,
  title={Classification with an edge: improving semantic image segmentation with boundary detection},
  author={Marmanis, Dimitrios and Schindler, Konrad and Wegner, Jan Dirk and Galliani, Silvano and Datcu, Mihai and Stilla, Uwe},
  journal={arXiv preprint arXiv:1612.01337},
  year={2016}
  }

 
