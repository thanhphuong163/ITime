# ITime: Multi-view Deep Markov Models for Time Series Anomaly Detection

This is the official repository of the paper Multi-view Deep Markov Models for Time Series Anomaly Detection [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10386155). The proposed model, called ITime, aims to find time steps in time series instances that have inconsistent features across multiple views.

## Install Python Environment

Make sure you have `miniconda` installed.

Create an virtual enviroment and install all packages from the requirements.txt file.

```bash
conda create --name mytorch python=3.8
conda activate mytorch
pip install -r requirements.txt
```

## Preprocess Datasets

- Create directories `data_preparation/raw_datasets` and `data_preparation/preprocessed_datasets`.
- Download the dataset to directory `raw_datasets`
  - DSA [link](https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities)
  - MEx [Link](https://archive.ics.uci.edu/dataset/500/mex)
  - MHealth [Link](http://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- Run each notebook file in directory `data_preparation/notebooks` for preprocessing each dataset.

## Run Model

Navigate to the `src_code` directory and run the shell script file `run_experiment.sh`.

## Cite
```latex
@inproceedings{nguyen2023multi,
  title={Multi-view Deep Markov Models for Time Series Anomaly Detection},
  author={Nguyen, Phuong and Tran, Hiep and Le, Tuan MV},
  booktitle={2023 IEEE International Conference on Big Data (BigData)},
  pages={799--808},
  year={2023},
  organization={IEEE}
}
```