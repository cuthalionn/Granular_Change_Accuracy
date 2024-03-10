# Granular Change Accuracy

Granular Change Accuracy official implementation.

## Installation
Clone the repository and change directory:
```bash
cd GCA
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

All figures from the paper can be generated using the notebooks in the `scripts` folder. The resulting figures will be saved under `analyses`.

## Citation

Coming soon...