# MACER: A Modular Framework for Accelerated Compilation Error Repair

This repository is the official implementation of MACER: A Modular Framework for Accelerated Compilation Error Repair. The paper describing the development of this framework can be accessed at [[this arXiv link]](https://arxiv.org/abs/2005.14015)

## Setup

### Installing Base Packages and Curate Datasets

To install all packages, and curate the datasets from TRACER and DeepFix, run

`sudo make install`

The MakeFile provided executes several steps. These are enumerated below for sake of clarity. If using the Makefile itself, you need not execute the following steps individually.

1. Ubuntu/Debian packages
    
    `sudo apt install clang python3-pip unzip gzip curl sqlite3`

1. Python packages
    
    `pip3 install --version -r requirements.txt`

1. Set Clang paths

    Create symbolic link to enable Python-Clang bind
    ```
    cd /usr/lib/x86_64-linux-gnu/
    sudo ln -s libclang-XX.YY.so.1 libclang.so
    ```

    Where, `XX.YY` is the version number of Clang installed on your system.

1. Pull TRACER's dataset, which is used for training and testing of MACER

    ```
    git clone https://github.com/umairzahmed/tracer.git
    unzip tracer/data/dataset/singleL/singleL_Test.zip -d tracer/data/dataset/singleL/
    unzip tracer/data/dataset/singleL/singleL_Train+Valid.zip -d tracer/data/dataset/singleL/
    ```

1. Repair class creation (refer Section 2. of our paper)

    `python3 -m srcT.DataStruct.ClusterError`

1. Pull DeepFix dataset, used for testing MACER

	```
    curl -O https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip
	unzip prutor-deepfix-09-12-2017.zip
	gzip -d prutor-deepfix-09-12-2017/prutor-deepfix-09-12-2017.db.gz
    ```

1. Extract DeepFix dataset into Macer's format

    ```
    sqlite3 -header -csv prutor-deepfix-09-12-2017/prutor-deepfix-09-12-2017.db "select * from Code where error<>'';" > prutor-deepfix-09-12-2017/deepfix_test.csv
    
	python3 -m srcT.DataStruct.PrepDeepFix
    ```
### Installing Python Library Dependencies

The MACER toolchain makes use of various standard libraries. A minimal list is provided in the [requirements.txt](requirements.txt) file. If you are a pip user, please use the following command to install these libraries:

```setup
pip3 install -r requirements.txt
```
**Note about version dependency**: although the requirements file specifies version dependencies to be exact, this is to err on the side of caution. For most of the libraries (e.g. `pandas` or `scikit-learn`), a more recent version should work well too. However, the dependency on the 1.2.4 version of the `edlib` library seems to be strict. Having a different version of this library may cause the toolchain to malfunction at evaluation time. Similarly, for `tensorflow`, it seems that version 2.0 can cause issues with training. We advise caution while using versions of libraries different from those mentioned in the requirements file.

## Training

To train the model(s) afresh on TRACER's training dataset consisting of single-line incorrect-correct C program pairs (please refer to Section 4 in the manuscript linked above), execute the command given below.

```train
python3 train.py 
```
Training might take between 7 minutes to 20 minutes depending on the machine configuration. Please note that executing the training routine will **overwrite any previously trained model**.

## Testing/Evaluation

### Repair individual C programs
To perform repair on a single C program using MACER, use
```eval
python3 testRepair.py <path-to-file.c> <PredK>
```

Example: an example program provided at [data/input/fig_1a.c](data/input/fig_1a.c) contains errors in a for loop statement specifier. To repair this program, execute the following command
```eval
$ python3 testRepair.py data/input/fig_1a.c 5
```

### Pred@k accuracy on TRACER's single-line test dataset
To obtain Pred@K accuracy on TRACER's test dataset (please refer to Table 4. in the manuscript linked above), execute the command given below. In the following, the parameter `PredK` specifies how many of the ranked suggestions by MACER should be considered (MACER offers, for any input program, a ranked list of repair suggestions). Please refer to Section 6.2 in the manuscript linked above for a description of this metric.

Evaluating Pred@K accuracy on all programs in the test dataset should take about 2-3 minutes depending on the machine configuration. Pred@K accuracy obtained will be printed at the end of execution.
```eval
python3 testPredAtK.py <PredK>
```                                     

Example:
```eval
$ python3 testPredAtK.py 5
Pred@5: 0.693
```

### Repair accuracy on TRACER's single-line test dataset
To obtain MACER's repair accuracy on TRACER's single-line test dataset (please refer to Table 4. in the manuscript linked above), execute the command given below. The parameter `PredK` continues to specify how many of the ranked suggestions by MACER should be considered. Please refer to Section 6.2 in the manuscript linked above for a description of this metric.

Evaluating repair accuracy on all programs in the test dataset might take 30 - 50 minutes depending on the machine configuration. The time taken is longer here since unlike Pred@K where the repaired program is merely checked for (abstract) match with the gold standard  (the students's own repaired program), here the program is recompiled. Repair accuracy obtained will be printed at the end of execution.

```eval
python3 testRepair.py tracer_single <PredK>
```

Example:
```eval
python3 testRepair.py tracer_single 5
```

### Repair accuracy on DeepFix's test dataset
To obtain MACER's repair accuracy on DeepFix's test dataset (please refer to Table 5. in the manuscript linked above), execute the command given below. The parameter `PredK` continues to specify how many of the ranked suggestions by MACER should be considered. 

Evaluating repair accuracy on all programs in the test dataset might take around 30 minutes depending on the machine configuration. Repair accuracy obtained will be printed at the end of execution.

```eval
python3 testRepair.py deepfix <PredK>
```

Example:
```eval
python3 testRepair.py deepfix 5
```

## Expected Results

MACER should offer the following performance on the given datasets. Minor deviations are expected if training afresh due to various randomizations used during the training process.

| Dataset            | Repair Accuracy |    
| ------------------ |---------------- | 
| TRACER SingleLine  |     0.805       |
| DeepFix            |     0.566       |

## Contributing
This repository is released under the MIT license. If you would like to submit a bugfix or an enhancement to MACER, please open an issue on this GitHub repository. We welcome other suggestions and comments too (please mail the corresponding author at purushot@cse.iitk.ac.in)
