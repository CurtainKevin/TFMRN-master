# Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures

We introduce LightTS, a light deep learning architecture merely based on simple MLP-based structures. The key idea of LightTS is to apply an MLP-based structure on top of two delicate down-sampling strategies, including interval sampling and continuous sampling, inspired by a crucial fact that down-sampling time series often preserves the majority of its information.

## Datasets

We conduct extensive experiments on eight widely used benchmark datasets.

### Overall information of the 8 datasets

| Datasets        | Variants | Timesteps | Granularity | Start time |
| :-------------- | -------- | --------- | ----------- | ---------- |
| ETTh1           | 7        | 17,420    | 1hour       | 7/1/2016   |
| ETTh2           | 7        | 17,420    | 1hour       | 7/1/2016   |
| ETTm1           | 7        | 69,680    | 15min       | 7/1/2016   |
| Weather         | 12       | 35,064    | 1hour       | 1/1/2010   |
| Traffic         | 862      | 17,544    | 1hour       | 1/1/2015   |
| Solar-Energy    | 137      | 52,560    | 1hour       | 1/1/2006   |
| Electricity/ECL | 321      | 26,304    | 1hour       | 1/1/2012   |
| Exchange-Rate   | 8        | 7,588     | 1hour       | 1/1/1990   |

Datasets are split into two folders, according to our experimental settings. 

```
./
└── datasets/
    ├── long
    │   ├── ETTh1.csv
    │   ├── ETTh2.csv
    │   └── ETTm1.csv
    |   ├── ECL.csv
    │   └── WTH.csv
    └── short
        ├── electricity.txt
        ├── exchange_rate.txt
        ├── solar_AL.txt
        └── traffic.txt
```

## Preparation

```
cd LightTS
conda create -n LightTS python=3.8
conda activate LightTS
pip install -r requirements.txt
```

## Training

### Scripts

You can check `./LightTS/scripts/` for all training scripts.
Use following command to train without hanging up.

```
cd LightTS
conda activate LightTS
sh ./scripts/run_*.sh
```
During and after training, you can find training outputs in \*.log files.

Before training, make sure that `chunk_size` is a divisor of `pred_len` or `horizon`.

### Long Sequence Forecasting

This part includes scripts `scripts/run_{ECL_long, ETT, weather}.sh`. If you want to test your own parameters, refer to one of the scripts.

In long sequence forecasting, we use `run_long.py`, `experiments/exp_long.py` and `data_process/long_dataloader.py`.

### Short Sequence Forecasting

This part includes scripts `scripts/run_{ECL_short, exchange, solar, traffic}.sh`. If you want to test your own parameters, refer to one of the scripts.

In long sequence forecasting, we use `run_short.py`, `experiments/exp_short.py` and `data_process/short_dataloader.py`.

### Ablation

This part explains script `run_ablation.sh`.

We only implemented ablation on short sequence forecasting datasets.











