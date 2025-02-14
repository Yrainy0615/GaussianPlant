# GaussianPlant: Structure-Aligned Gaussian Splatting for Plant Structure ExtractionYang Yang, Fumio Okura<br>
| [Webpage](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) | [Full Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) | <br>
This repository contains the official authors implementation associated with the paper "GaussianPlant: Structure-Aligned Gaussian Splatting for Plant Structure Extraction", which can be found [here](https://github.com/Yrainy0615/GaussianPlant.git). <br>
<a href="http://cvl.ist.osaka-u.ac.jp/en/"><img height="100" src="assets/osaka_logo.png"> </a>


Abstract: *We present a method for jointly recovering the appearance and structural organization of plants from multi-view images using 3D Gaussian Splatting (3DGS). While existing 3DGS approaches prioritize appearance fidelity or surface-aligned representations, they do not explicitly capture the underlying structure of objects. Our method introduces structural primitives (StPrs), initialized from clustered SfM points and represented as 3D Gaussians. These StPrs are first optimized to capture the coarse structural form of the plant, after which appearance Gaussians (AppGS) are bound to StPrs and jointly optimized through our proposed optimization strategy. The final plant structure is extracted from the optimized StPrs.
Our approach enables structure-aware 3DGS without requiring predefined skeleton priors or parametric templates. Experimental results demonstrate that our method effectively reconstructs both appearance and structure of plants, highlighting the potential of 3DGS as a framework for structural information extraction beyond scene representation.*

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>


## Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
# SSH
git clone git@github.com:Yrainy0615/GaussianPlant.git --recursive
```
or
```shell
# HTTPS
git clone https://github.com/Yrainy0615/GaussianPlant.git --recursive
```

## Overview

The codebase has 4 main components:
- A PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs
- A network viewer that allows to connect to and visualize the optimization process
- An OpenGL-based real-time viewer to render trained models in real-time.
- A script to help you turn your own images into optimization-ready SfM data sets

The components have different requirements w.r.t. both hardware and software. They have been tested on Windows 10 and Ubuntu Linux 24.04. Instructions for setting up and running each of them are found in the sections below.




