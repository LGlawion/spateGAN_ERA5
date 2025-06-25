## spateGAN_ERA5

**spateGAN-ERA5** is a deep learning framework designed for the spatio-temporal downscaling of ERA5 precipitation data. Utilizing a probabilistic conditional Generative Adversarial Networks (cGANs), it enhances the resolution of ERA5 data from 24 km and 1-hour intervals to 2 km and 10-minute intervals. This advancement enables high-resolution rainfall fields with realistic spatio-temporal patterns and accurate rain rate distributions, including extreme events. (https://www.nature.com/articles/s41612-025-01103-y)


---

### Features

- **Global Generalization**  
  Trained on German rain gauge adjusted radar data and validated in regions like the US and Australia.

- **High-Resolution Output**  
  Converts coarse ERA5 precipitation data into 2 km × 10 min fields, essential for flood and hydrological risk assessments.

- **Uncertainty Quantification**  
  Generates diverse ensembles of precipitation scenarios.

---

### Getting Started
#### Installation

```bash
mamba env create -f environment.yaml
or
conda env create -f environment.yaml
```

#### Example: Downscaling Over Germany

A demonstration notebook is included in the repo:

downscaling.ipynb — uses a small sample dataset to show how to apply spateGAN-ERA5 for downscaling ERA5 precipitation over Germany.

The notebook includes:

- Loading preprocessed ERA5 input samples

- Running the pre-trained spateGAN-ERA5 model

- Visualizing and comparison of high-resolution rain field predictions to rainfall observations



### License and co

The content of this repository is released under the terms of the MIT license.

Consider citing if you use spateGAN-ERA5:
<pre>
@article{glawion_global_2025,
	title = {Global spatio-temporal {ERA5} precipitation downscaling to km and sub-hourly scale using generative {AI}},
	doi = {10.1038/s41612-025-01103-y},
	journal = {npj Climate and Atmospheric Science},
	author = {Glawion, Luca and Polz, Julius and Kunstmann, Harald and Fersch, Benjamin and Chwala, Christian},
	year = {2025},
}
</pre>
