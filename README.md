# Automatic data frame processing for AutoML

Environment for knowledge analysis - WEKA:
https://github.com/Waikato/weka

AutoML - AutoWEKA:
https://github.com/automl/autoweka

Dataset - vertical farming:
https://www.kaggle.com/datasets/midouazerty/work-for-parmavir

## Description and setting dataset

The dataset contains information about sections called cubes that are divided into two layers `A` and `B`.
The presented values ​​correspond to the temperature and humidity values ​​in the sections.
Using `python` tools, in particular `pandas`, load the dataset.

```python
import pandas as pd

df = pd.read_csv('../data/cubes.csv', low_memory=False)
df
```

|        |   Unnamed: 0 |   Cube ID |   Timestamp | Temperature Layer A   | Temperature Layer B   |   Door | Humidity Layer A       | Humidity Layer B       |
|-------:|-------------:|----------:|------------:|:----------------------|:----------------------|-------:|:-----------------------|:-----------------------|
|      0 |            0 |        49 |  1451606401 | nan                   | nan                   |      0 | nan                    | nan                    |
|      1 |            1 |        95 |  1451606402 | nan                   | 25.711898671792053°C  |    nan | nan                    | nan                    |
|      2 |            2 |        48 |  1451606402 | nan                   | nan                   |    nan | nan                    | nan                    |
|      3 |            3 |        55 |  1451606402 | nan                   | nan                   |    nan | nan                    | 8.594411333817883g/m3  |
|      4 |            4 |        90 |  1451606403 | nan                   | nan                   |    nan | nan                    | nan                    |
|      5 |            5 |         5 |  1451606406 | nan                   | nan                   |      0 | nan                    | nan                    |
|      6 |            6 |        73 |  1451606407 | 23.776610608962223°C  | nan                   |    nan | nan                    | nan                    |
...
| 400284 |       400284 |         6 |  1455929825 | nan                   | nan                   |    nan | nan                    | nan                    |
| 400285 |       400285 |        36 |  1455929967 | nan                   | nan                   |    nan | nan                    | nan                    |
| 400286 |       400286 |         0 |  1455929993 | nan                   | nan                   |    nan | nan                    | nan                    |

```
400287 rows × 8 columns
```

There are two identical `index` columns in the table, get rid of one.

```python
df = df.drop(columns=['Unnamed: 0'])
```

For convenience, shift the `Timestamp` to zero.

```python
df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min()
```

For classification and regression problems, only real numbers are needed, reduce the `Temperature` and `Humidity` columns to numbers.

```python
df['Temperature Layer A'] = df['Temperature Layer A'].str.replace('°C', '')
df['Temperature Layer B'] = df['Temperature Layer B'].str.replace('°C', '')

df['Humidity Layer A'] = df['Humidity Layer A'].str.replace('g/m3', '')
df['Humidity Layer B'] = df['Humidity Layer B'].str.replace('g/m3', '')
```

```python
df
```

|        |   Cube ID |   Timestamp |   Temperature Layer A |   Temperature Layer B |   Door |   Humidity Layer A |   Humidity Layer B |
|-------:|----------:|------------:|----------------------:|----------------------:|-------:|-------------------:|-------------------:|
|      0 |        49 |           0 |              nan      |              nan      |      0 |          nan       |          nan       |
|      1 |        95 |           1 |              nan      |               25.7119 |    nan |          nan       |          nan       |
|      2 |        48 |           1 |              nan      |              nan      |    nan |          nan       |          nan       |
|      3 |        55 |           1 |              nan      |              nan      |    nan |          nan       |            8.59441 |
|      4 |        90 |           2 |              nan      |              nan      |    nan |          nan       |          nan       |
|      5 |         5 |           5 |              nan      |              nan      |      0 |          nan       |          nan       |
|      6 |        73 |           6 |               23.7766 |              nan      |    nan |          nan       |          nan       |
...
| 400284 |         6 |     4323424 |              nan      |              nan      |    nan |          nan       |          nan       |
| 400285 |        36 |     4323566 |              nan      |              nan      |    nan |          nan       |          nan       |
| 400286 |         0 |     4323592 |              nan      |              nan      |    nan |          nan       |          nan       |

```
400287 rows × 7 columns
```

![plot](./img/hist1.jpg)

For a given sample, it is possible to predict the `Temperature` and `Humidity` for the cubes over time.
It is necessary to find out the importance of the `Door` feature on the influence of the model,
so let's compare the results of AutoWEKA for samples with and without the `Door` feature.

#### Temperature Layer A w/ Door
| N | params        | EER    | training time  | CC     | MAE    | RMSE   | RAE, %  | RRSE, % | Total of cfg | best classifier   |
| - | ------------- | ------ | -------------- | ------ | ------ | ------ | ------- | ------- | ------------ | ----------------- |
| 1 | 15m 1024MB 1r | 1.2625 |  5.580s        | 0.8621 | 0.9437 | 1.2626 | 45.6575 | 51.5778 | 26           | AttributeSelected |
| 2 | 30m 1536MB 2r | 0.6202 | 28.701s        | 0.9782 | 0.3832 | 0.5153 | 18.5402 | 21.0499 | 43           | RandomForest      |
| 3 | 45m 2048MB 4r | 0.6202 | 29.202s        | 0.9782 | 0.3832 | 0.5153 | 18.5402 | 21.0499 | 107          | RandomForest      |

#### Temperature Layer B w/ Door
| N | params        | EER    | training time  | CC     | MAE    | RMSE   | RAE, %  | RRSE, % | Total of cfg | best classifier   |
| - | ------------- | ------ | -------------- | ------ | ------ | ------ | ------- | ------- | ------------ | ----------------- |
| 1 | 15m 1024MB 1r | 1.2303 |  5.134s        | 0.8655 | 0.9249 | 1.2305 | 44.8119 | 50.9741 | 29           | AttributeSelected |
| 2 | 30m 1536MB 2r | 1.1323 |  4.538s        | 0.8832 | 0.8903 | 1.1324 | 43.1376 | 46.9077 | 60           | REPTree           |
| 3 | 45m 2048MB 4r | 0.6280 | 28.213s        | 0.9782 | 0.3776 | 0.5080 | 18.2964 | 21.0446 | 126          | RandomForest      |

#### Humidity Layer A w/ Door
| N | params        | EER    | training time  | CC     | MAE    | RMSE   | RAE, %  | RRSE, % | Total of cfg | best classifier   |
| - | ------------- | ------ | -------------- | ------ | ------ | ------ | ------- | ------- | ------------ | ----------------- |
| 1 | 15m 1024MB 1r | 0.9325 |  0.173s        | 0.9554 | 0.7384 | 0.9326 | 78.8383 | 29.5282 | 92           | IBk               |
| 2 | 30m 1536MB 2r | 0.8671 |  0.203s        | 0.9614 | 0.6731 | 0.8695 | 71.8692 | 27.5321 | 186          | REPTree           |
| 3 | 45m 2048MB 4r | 0.4839 |  3.419s        | 0.9901 | 0.3315 | 0.4838 | 35.3939 | 15.3191 | 272          | RandomForest      |

#### Humidity Layer B w/ Door
| N | params        | EER    | training time  | CC     | MAE    | RMSE   | RAE, %  | RRSE, % | Total of cfg | best classifier   |
| - | ------------- | ------ | -------------- | ------ | ------ | ------ | ------- | ------- | ------------ | ----------------- |
| 1 | 15m 1024MB 1r | 0.6582 |  9.362s        | 0.9791 | 0.4554 | 0.6624 | 47.3593 | 20.8447 | 89           | RandomForest      |
| 2 | 30m 1536MB 2r | 0.6582 | 10.258s        | 0.9791 | 0.4554 | 0.6624 | 47.3593 | 20.8447 | 171          | RandomForest      |
| 3 | 45m 2048MB 4r | 0.6582 |  9.614s        | 0.9791 | 0.4554 | 0.6624 | 47.3593 | 20.8447 | 268          | RandomForest      |