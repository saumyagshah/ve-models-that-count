# ve-models-that-count
 Code for Object Detection-Based Count Module (OCM) for 16824: Visual Recognition Project
 
 Instructions for running the code for OCM:
 - Set the paths in the `get_ve_loader` function of `data_grid_ve.py` for initializing `snli_dataset`.
 - Run `train_grid_ve_arch_change.py` for training the model with the counting module and `train_grid_noc_ve_arch_change.py` for training the model without the counting module

 Instructions for running the code for ZCM:
 - Set the paths in the `get_ve_loader` function of `data_ve.py` for initializing `snli_dataset`.
 - Run `train_ve_arch_change.py` for training the model with the counting module and `train_ve_no_counting.py` for training the model without the counting module
