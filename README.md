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

### Datasets

Training, model selection and evaluation datasets are available at https://doi.org/10.5281/zenodo.17417589

To run the model for downscaling precipitation data in inference, ERA5 data in hourly resolution can be downloaded
[here](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
as a netCDF file.
The variables "Convective precipitation" and "Large scale precipitation", a minimal spatial extent of 672kmx672km and a minimal time sequence of 16 hrs are required.
 
---

### Getting Started
#### Installation

**conda**
```bash
mamba env create -f environment.yaml
or
conda env create -f environment.yaml
```

**uv**
clone repo
install uv: https://docs.astral.sh/uv/getting-started/installation/
```bash
cd spateGAN_ERA5
uv sync #creates .venv
```
use `spateGAN_ERA5/.venv/bin/python3` as your interpreter

#### Downscaling
Update ```config.yaml``` to your downscaling domain (defined as centre lat lon coordinates), input path of ERA5 netCDF and output path.

Run
```bash
uv run main.py
``` 
High resolution precipitation fields are saved in UTM projection (2km, 10min resolution) and lat lon (0.018° and 10min resolution).

#### Probabilistic downscaling
For probabilistic downscaling ```seed``` and ```slide```parameter in ```config.yaml```can be changed.

#### Evaluation example
As an evaluation example, a notebook is included in the repo:

downscaling.ipynb (and downscaling_example.py) — uses a small sample dataset to show spateGAN-ERA5 for downscaling ERA5 precipitation over Germany and compare it to radar observations.


### License and co

The content of this repository is released under the terms of the MIT license.

Consider citing if you use spateGAN-ERA5:
<pre>
@article{spateGAN_era5_2025,
	title = {Global spatio-temporal {ERA5} precipitation downscaling to km and sub-hourly scale using generative {AI}},
	doi = {10.1038/s41612-025-01103-y},
	journal = {npj Climate and Atmospheric Science},
	author = {Glawion, Luca and Polz, Julius and Kunstmann, Harald and Fersch, Benjamin and Chwala, Christian},
	year = {2025},
}
</pre>
