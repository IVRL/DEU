# Denoising experiments
## pre_trained
* `model`:the pre_trained denoisers
  * `DnCNN`:pre_trained denoised model for gray images
  * `MemNet`:pre_trained denoised model for gray images
  * `RIDNet`:pre_trained denoised model for gray images
  * `DnCNN_color`:pre_trained denoised model for color images
  
* `saved_models`:the saved fusion models (with all augmentation modes) for each pre_trained denoisers
  * `DnCNN` [net_F](https://github.com/IVRL/DEU/tree/master/Denoise_Fusion/saved_models/DnCNN/net_F) : fusion models for gray images pre-denoised by DnCNN  for special noise levels
  * `MemNet` [net_F](https://github.com/IVRL/DEU/tree/master/Denoise_Fusion/saved_models/MemNet/net_F) : fusion models for gray images pre-denoised by MemNet for special noise levels
  * `RIDNet` [net_F](https://github.com/IVRL/DEU/tree/master/Denoise_Fusion/saved_models/RIDNet/net_F) : fusion models for gray images pre-denoised by RIDNet for special noise levels
  * `DnCNN_color` [net_F](https://github.com/IVRL/DEU/tree/master/Denoise_Fusion/saved_models/DnCNN_color/net_F) : fusion models for color images pre-denoised by DnCNN_color for special noise levels

* `test.py`:test for the saved fusion models:

```python test.py --denoise_net DnCNN --color_mode gray ```






## Re-training
To reproduce the regular training of Fusion models for gray images (net_mode DnCNN/MemNet/RIDNet respectively)

```python train.py --denoise_net DnCNN ```

To reproduce the regular training of Fusion models for gray images for different modes:

For flip and rotation:

```python train.py --mode_list 0 1 2 3 4 5 6 7  ```

for DCT mask:

```python train.py --mode_list 0 8 9 10 11 12  ```

for all(default):

```python train.py --mode_list 0 1 2 3 4 5 6 7 8 9 10 11 12  ```

To reproduce the regular training of Fusion models for color images (net_mode DnCNN_color)

```python train.py --denoise_net DnCNN_color --color_mode color ```

To reproduce the regular training of channel attention or spatial attention models or fusion models for gray images (ensemble_method S(spatial attention)/C(channel attention)/F(fusion))

```python train.py --net_mode DnCNN --ensemble_method S ```

The spatial attention and channel attention are just fit for gray models until now.

The default setting is the fusion models for gray images with all augmentaion modes pre denoised by DnCNN 

```python train.py  ```





