# Appropriate Answer Prediction

 (科技大擂台 與AI對話)
 
----

## Feature

* A CNTK (Microsoft deep learning toolkit) implementation of CS565600 competition
* We use LSTM + attention to do this task
* For more model information, please refer to the [report](https://github.com/zlsh80826/AppropriateResponsePrediction/blob/master/script/Report.ipynb)
* If you meet any problem in this repo, feel free to contact zlsh80826@gmail.com

## Requirements

Here are some required libraries for training

### General
* python3
* cuda-9.0 (CNTK required)
* openmpi-1.10 (CNTK required)
* gcc >= 6 (CNTK required)

### Python
* Please refer requirements.txt

## Usage 

We recommand you to run all the scripts in script directory

```Bash
cd AppropriateResponsePrediction/script
```

Each script contain helper, you can check it for customed settings.

```Bash
python <some script>.py --help
``` 

### Preprocess

This script will convert the text format program to processed `npy` format.

You can specify `--threads` to indicate how many threads you want to use.

```Bash
python preprocessing.py
```

### Train Fasttext

This script will train the Traditional Chinese Embedding with processed data.

```Bash
python train_fasttext.py
```

### Generate The Training Data

Default settings will generate 4 million training data, which will consume about 8 GB disk space.

```Bash
python gen_training.py
```

### Convert tsv to ctf

CNTK support large training file, but we need to convert it to ctf format.

```Bash
python tsv2ctf.py
```

### Train

Default settings will run 300 epochs and save the checkpoint of each epoch.

```Bash
python train.py
```

### Inference

Inference script will read the checkpoint and do the inference. So you can inference while training, 4 - 10 epochs result is good enough in my experimence.

```Bash
python inference.py
```

### Performance

Based on the [Kaggle Leaderboard](https://www.kaggle.com/c/datalabcup-predicting-appropriate-response/leaderboard), our implementation is second prize (the first two are fake).

|              |Public score|Private score|
|--------------|------------|-------------|
|Single Model  | 73.6       |  71.6       |
|Ensemble Model| 76.4       |  72         |
