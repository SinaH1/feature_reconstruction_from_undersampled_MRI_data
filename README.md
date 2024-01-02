## Masterthesis - Sina Heizmann

# Feature reconstruction from undersampled MRI data using Neural Networks

The implementation of the VarNet is based on the GitHub repository [pytorch_mri_variationalnetwork](https://github.com/rixez/pytorch_mri_variationalnetwork). References for the Variational Network (VarNet) are
- Hammernik et al., [Learning a variational network for reconstruction of accelerated MRI data](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.26977), Magnetic Resonance in Medicine, 79(6), pp. 3055-3071, 2018.
- Knoll et al., [Assessment of the generalization of learned image reconstruction and the potential for transfer learning](https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.27355), Magnetic Resonance in Medicine, 2018 (early view).


The implementation of the U-Net architecture is based on the GitHub repository [Pytorch-UNet/unet/](https://github.com/milesial/Pytorch-UNet/tree/master/unet). Reference for the U-Net is
- Ronneberger et al., [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18, pp. 234–241, 2015.



## 6.1. Neglection of the correction term
Values of Section 6.1. can be directly calculated with the following run in the console:
```
python reconstruction_and_analysis/_correction_term.py --operator {operator} --task {task}
```
Choose between:
- **operator:**	 TF, FD, Convolution
- **task:**		 r_coil_approximation_residual, e_correction_term, d_landweber, d_CG


## 6.2. Feature reconstruction with VarNet
Original image (net_type=original), partial derivatives (net_type=only_dx/only_dy), simultaneous convolutional feature (net_type=fixed_kernels) reconstructions with VarNet are saved in 'results/adjusted_Landweber/test' with the following run in the console:
```
python adjusted_VarNet/run_varnet.py --net_type {net_type} --operator {operator} --mode {mode} --epoch {number of epochs}
```
Choose between:
- **net_type:**	 original, only_dx, only_dy, fixed_kernels
- **operator:**	 TF, FD
- **mode:**	 train, eval

Other choices are possible for a different network architecture or training settings. Choices are listed by:
```
python adjusted_VarNet/run_varnet.py -h
```


## 6.3. Combining image reconstruction and edge detection
Comparison of direct approach and the method in two steps for edge detection can be observed in 'results/edge_detection' when executing:
1. Landweber method:
```
python reconstruction_and_analysis/_landweber_best_per_partial.py  --operator {operator} --show_images {bool} 
```
2. CG method:
```
python reconstruction_and_analysis/_CG_best_per_partial.py --operator {operator} --show_images {bool} 
```
3. VarNet:
```
python adjusted_VarNet/run_varnet.py --net_type {net_type} --operator {operator} --mode {mode} --epoch {number of epochs}

```
Choose between:
- **operator:**	 TF, FD
- **net_type:**	 original, only_dx, only_dy, fixed_kernels
- **mode:**	 train, eval

Other choices are possible for a different network architecture or training settings. Choices are listed by:
```
python adjusted_VarNet/run_varnet.py -h
```


## 6.4. Concatenation of image reconstruction and segmentation
### Separate Training
#### VarNet
```
python adjusted_VarNet/run_varnet.py --net_type {net_type} --operator {operator} --mode {mode} --epoch {number of epochs}

```
Choose between:
- **net_type:**	 original, only_dx, only_dy, fixed_kernels
- **operator:**	 TF, FD
- **mode:**	 train, eval
Other choices are possible for a different network architecture or training settings. Choices are listed by:
```
python adjusted_VarNet/run_varnet.py -h
```

#### UNet
```
python UNet/run_unet.py --net_type {net_type} --mode {mode} --epoch {number of epochs}

```
Choose between:
- **net_type:**	 original, orig+dx, learned_kernels, orig+kernel
- **mode:**	 train, eval
Other choices are possible for a different network architecture or training settings. Choices are listed by:
```
python UNet/run_unet.py -h
```



### Concatenation image reconstruction and segmentation
```
python concatenated_networks/run_networks.py --net_type {net_type} --mode {mode} --epoch {number of epochs}

```
Choose between:
- **net_type:**	 original, learned_kernels
- **mode:**	 train, eval_varnet, eval_seg
Other choices are possible for a different network architecture or training settings. Choices are listed by:
```
python concatenated_networks/run_networks.py -h
```


### Concatenation feature reconstruction and segmentation
With the network type “1_kernel”, the convolutional feature reconstruction with one kernel is executed.
With the network type “learned_kernels”, the reconstruction of the original image and three learned features is executed.
```
python concatenated_networks/run_networks_alternate_training.py --net_type {net_type} --mode {mode} --epoch {number of epochs}

```
Choose between:
- **net_type:**	 1_kernel, learned_kernels
- **mode:**	 train, eval_varnet, eval_seg
Other choices are possible for a different network architecture or training settings. Choices are listed by:
```
python concatenated_networks/run_networks_alternate_training.py -h
```






