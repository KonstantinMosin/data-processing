# Automatic data frame processing for AutoML

Environment for knowledge analysis - WEKA:
https://github.com/Waikato/weka

AutoML - AutoWEKA:
https://github.com/automl/autoweka

Dataset - vertical farming:
https://www.kaggle.com/datasets/midouazerty/work-for-parmavir

## Iteration 1. Manual dataset configuration, launching AutoWEKA, comparing the results.

load the dataset and see the type of features
```python
import pandas as pd

df = pd.read_csv('../data/cubes.csv', low_memory=False)
df.dtypes
```
```
Unnamed: 0               int64
Cube ID                  int64
Timestamp                int64
Temperature Layer A     object
Temperature Layer B     object
Door                   float64
Humidity Layer A       float64
Humidity Layer B       float64
dtype: object
```


columns `Unnamed: 0` and `Door` are not interesting

```python
df = df.drop(columns=['Unnamed: 0','Door']) # exclude index and unnecessary columns
```


move `Timestamp` to `0` and cast `Temperature Layer A` and `Temperature Layer B` to float

```python
df['Timestamp'] = df['Timestamp'] - df['Timestamp'].min() # move the timer to the start

df['Temperature Layer A'] = df['Temperature Layer A'].str.replace('°C', '').astype(float)
df['Temperature Layer B'] = df['Temperature Layer B'].str.replace('°C', '').astype(float)
# cast data to type numeric

df['Humidity Layer A'] = df['Humidity Layer A']
df['Humidity Layer B'] = df['Humidity Layer B']
```


delete rows that do not contain any information

```python
df = df.dropna(how='all', subset=[
    'Temperature Layer A',
    'Temperature Layer B',
    'Humidity Layer A',
    'Humidity Layer B'
    ])
# clear data from empty rows
```


build histograms

```python
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

bins = 100

fig, axs = plt.subplots(2, 2)

axs[0, 0].hist(df['Temperature Layer A'].values, bins=bins)
axs[0, 0].set_title('Temperature Layer A')

axs[0, 1].hist(df['Temperature Layer B'].values, bins=bins)
axs[0, 1].set_title('Temperature Layer B')

axs[1, 0].hist(df['Humidity Layer A'].values, bins=bins)
axs[1, 0].set_title('Humidity Layer A')

axs[1, 1].hist(df['Humidity Layer B'].values, bins=bins)
axs[1, 1].set_title('Humidity Layer B')

plt.show()
```

![plot](./img/hist1.jpg)


remove from dataset outliers and look at the result

```python
import numpy as np

df['Humidity Layer A'] = [x if x < 25 else np.nan for x in df['Humidity Layer A']]
df['Humidity Layer B'] = [x if x < 25 else np.nan for x in df['Humidity Layer B']]
# clear humidity from outliers
```

![plot](./img/hist2.jpg)


| feature             | launch parameters*    | estimated error  | metrics**                                            | total of configuration | result***          |
|:------------------- | --------------------- |:----------------:| ---------------------------------------------------- |:----------------------:|:------------------:|
| Temperature layer A | 5 min<br/>1024MB<br/>1 run  | 0.56024657       | 0.9781<br/>0.3842<br/>0.5157<br/>18.5953<br/>21.1105 | 18       | trees.RandomForest |
| Temperature layer A | 15 min<br/>2048MB<br/>4 run | 0.56024657       | 0.9781<br/>0.3842<br/>0.5157<br/>18.5953<br/>21.1105 | 89       | trees.RandomForest |
| Temperature layer B | 5 min<br/>1024MB<br/>1 run  | 0.57685781       | 0.9778<br/>0.3794<br/>0.5131<br/>18.2875<br/>21.1949 | 23       | trees.RandomForest |
| Temperature layer B | 15 min<br/>2048MB<br/>4 run | 0.57685781       | 0.9778<br/>0.3794<br/>0.5131<br/>18.2875<br/>21.1949 | 23       | trees.RandomForest |
| Humidity layer A    | 5 min<br/>1024MB<br/>1 run  | 0.98145939       | 0.3704<br/>0.7756<br/>0.9815<br/>92.7262<br/>92.8883 | 58       | trees.M5P          |
| Humidity layer A    | 15 min<br/>2048MB<br/>4 run | 0.93240144       | 0.4704<br/>0.7382<br/>0.9324<br/>88.2446<br/>88.2453 | 177      | trees.RandomTree   |
| Humidity layer B    | 5 min<br/>1024MB<br/>1 run  | 0.97352648       | 0.4358<br/>0.7695<br/>0.9735<br/>89.5807<br/>90.1227 | 49       | trees.M5P          |
| Humidity layer B    | 15 min<br/>2048MB<br/>4 run | 0.94186592       | 0.4897<br/>0.7469<br/>0.9419<br/>86.9592<br/>87.1918 | 182      | trees.M5P          |

\* CLI: `-timeLimit x -memLimit y -parallelRund z`

** metrics: correlation coefficient, mean absolute error, root mean squared error, reative squared error %, root relative squared error %

*** best classifier
