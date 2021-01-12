# Denoising experiments
## Pre-trained
We provide pre-trained fusion models:

* `saved_models`: the saved fusion models (with all augmentation modes) for each pre-trained denoiser
  * `DnCNN` [net_F](https://github.com/IVRL/DEU/tree/main/Denoise_Fusion/saved_models/DnCNN/net_F) : fusion models for grayscale images with the DnCNN backbone for different noise levels.
  * `MemNet` [net_F](https://github.com/IVRL/DEU/tree/main/Denoise_Fusion/saved_models/MemNet/net_F) : fusion models for grayscale images with the MemNet backbone for different noise levels.
  * `RIDNet` [net_F](https://github.com/IVRL/DEU/tree/main/Denoise_Fusion/saved_models/RIDNet/net_F) : fusion models for grayscale images with the RIDNet backbone for different noise levels.
  * `DnCNN_color` [net_F](https://github.com/IVRL/DEU/tree/main/Denoise_Fusion/saved_models/DnCNN_color/net_F) : fusion models for color images with the DnCNN_color backbone for different noise levels.

* `test.py`: test for the saved fusion models

To test the fusion models for grayscale images (denoise_net DnCNN/MemNet/RIDNet respectively):

```python test.py --denoise_net DnCNN --color_mode gray ```

To test the fusion models for color images (denoise_net DnCNN_color):

```python test.py --denoise_net DnCNN_color --color_mode color ```




## Re-training
The default settings give the fusion model for grayscale images with all augmentaion modes, with the DnCNN backbone:

```python train.py  ```

To reproduce the regular training of Fusion models for grayscale images (denoise_net DnCNN/MemNet/RIDNet respectively):

```python train.py --denoise_net DnCNN ```

To reproduce the regular training of Fusion models for grayscale images for different modes,

- for flip and rotation:

```python train.py --mode_list 0 1 2 3 4 5 6 7  ```

- for DCT mask:

```python train.py --mode_list 0 8 9 10 11 12  ```

- for all (default):

```python train.py --mode_list 0 1 2 3 4 5 6 7 8 9 10 11 12  ```

To reproduce the regular training of Fusion models for color images (net_mode DnCNN_color):

```python train.py --denoise_net DnCNN_color --color_mode color ```

To reproduce the regular training of channel attention or spatial attention models or fusion models for grayscale images (ensemble_method S(spatial attention)/C(channel attention)/F(fusion)):

```python train.py --net_mode DnCNN --ensemble_method S ```

PS: The spatial attention and channel attention ablation models are only fit for grayscale denoising until now.
