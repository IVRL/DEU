# Denoising experiments
## pre_trained
* `model`:the pre_trained denoisers
  * `DnCNN`:pre_trained denoised model for gray images
  * `MemNet`:pre_trained denoised model for gray images
  * `RIDNet`:pre_trained denoised model for gray images
  * `DnCNN_color`:pre_trained denoised model for color images
  
* `saved_models`:the saved models for each pre_trained denoisers
  * `DnCNN` `F` `Joint` : fusion models for gray images pre-denoised by DnCNN  for all manipulation modes
  * `MemNet` `F` `Joint`: fusion models for gray images pre-denoised by MemNet for all manipulation modes
  * `RIDNet` `F` `Joint`: fusion models for gray images pre-denoised by RIDNet for all manipulation modes
  * `DnCNN_color` `F` `Joint` : fusion models for color images pre-denoised by DnCNN_color for all manipulation modes
  
  * except for the fusion models with joint manipulation modes, we also provide other saved models for channel attention module (C) and spatial attention module(S) with Spatial domain manipulation modes(SM）or frequency domain manipulation modes(FM),rather than Joint, which are shown on 

* `test.py`:test for the saved fusion models:  

If you want to test the fusion model for DnCNN with joint manipulation modes

```python test.py --denoise_net DnCNN --color_mode gray ```






## Re-training
To reproduce the regular training of Fusion models for gray images (net_mode DnCNN/MemNet/RIDNet respectively)

```python train.py --denoise_net DnCNN ```

To reproduce the regular training of models for gray images for different manipulation modes:

For flip and rotation:

```python train.py --manipulation_mode SM  ```

for DCT mask:

```python train.py --manipulation_mode FM  ```

for all(default):

```python train.py --manipulation_mode Joint  ```

To reproduce the regular training of Fusion models for color images (net_mode DnCNN_color)

```python train.py --denoise_net DnCNN_color --color_mode color ```

To reproduce the regular training of channel attention or spatial attention models or fusion models for gray images (ensemble_method S(spatial attention)/C(channel attention)/F(fusion))

```python train.py --ensemble_method S ```

To reproduce the the regular training of fusion models with all manipulation modes(Joint) for pre_trained net DnCNN:

```python train.py  --denoise_net DnCNN --ensemble_method F  --manipulation_mode Joint```

The spatial attention and channel attention are just fit for gray models until now.

The default setting is the fusion models for gray images with all augmentaion modes pre denoised by DnCNN 

```python train.py  ```





