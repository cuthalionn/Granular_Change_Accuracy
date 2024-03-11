# Granular Change Accuracy

Granular Change Accuracy official implementation.

## Installation and Setup
Clone the repository and change directory:
```bash
cd Granular_Change_Accuracy
```
Then create new environment and install dependencies:

```bash
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
pip install -e . 
```

## Usage
Change directory into the main folder in the repository 
```bash
cd GCA
python src/compute_accuracies.py "path_to_data_folder"
```

Check [data](data/) folder for data format sample.

## Additional Figures

Download `data.json` and `test_dials.json` for Multiwoz from this [link](https://drive.google.com/drive/folders/1B6tLIfQh31OcujmyPPzIwCPdaq7ZcEt9?usp=sharing) to `data` folder before starting analysis.

All figures from the paper can be generated using the notebooks in the `scripts` folder. The resulting figures will be saved under `analyses`.

## Citation

Coming soon...
