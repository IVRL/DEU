# Deep Gaussian Denoiser Epistemic Uncertainty and Decoupled Dual-Attention Fusion

**Authors**: Xiaoqi Ma, Xiaoyu Lin, [Majed El Helou](https://majedelhelou.github.io/), and Sabine Süsstrunk

![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![pytorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic)
![CUDA 10.1](https://camo.githubusercontent.com/5e1f2e59c9910aa4426791d95a714f1c90679f5a/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f637564612d31302e312d677265656e2e7376673f7374796c653d706c6173746963)

{Note: paper under submission}

#### [[Paper]](404)

> **Abstract:** *Following the performance breakthrough of denoising networks, improvements have come chiefly through novel architecture designs and increased depth. While novel denoising networks were designed for real images coming from different distributions, or for specific applications, comparatively small improvement was achieved on Gaussian denoising. The denoising solutions suffer from epistemic uncertainty that can limit further advancements. This uncertainty is traditionally mitigated through different ensemble approaches. However, such ensembles are prohibitively costly with deep networks, which are already large in size.*
>
> *Our work focuses on pushing the performance limits of state-of-the-art methods on Gaussian denoising. We propose a model-agnostic approach for reducing epistemic uncertainty while using only a single pretrained network. We achieve this by tapping into the epistemic uncertainty through augmented and frequency-manipulated images to obtain denoised images with varying error. We propose an ensemble method with two decoupled attention paths, over the pixel domain and over that of our different manipulations, to learn the final fusion. Our results significantly improve over the state-of-the-art baselines and across varying noise levels.*


## Code overview
All the models and instructions for testing with our pretrained networks, and for retraining, are detailed in the [Denoise_Fusion](https://github.com/IVRL/DEU/tree/main/Denoise_Fusion) directory.

## Citation
```bibtex
@article{ma2021deep,
    title   = {Deep {Gaussian} Denoiser Epistemic Uncertainty and Decoupled Dual-Attention Fusion},
    author  = {Ma, Xiaoqi and Lin, Xiaoyu and El Helou, Majed and S{\"u}sstrunk, Sabine},
    journal = {arXiv preprint},
    year    = {2021}
}
```
