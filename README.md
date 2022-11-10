# Automatic data frame processing for AutoML

Environment for knowledge analysis - WEKA:
https://github.com/Waikato/weka

AutoML - AutoWEKA:
https://github.com/automl/autoweka

Dataset - vertical farming:
https://www.kaggle.com/datasets/midouazerty/work-for-parmavir

## Iteration 1. Manual dataset configuration, launching AutoWEKA, comparing the results.

### Data preprocessing

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

### Work with AutoWEKA

system: AMD Ryzen 3 2200U, 4GB RAM

version: WEKA 3.8.6, AutoWEKA 2.6.4

Take for consideration 4 feature: `Temperature layer A`, `Temperature layer B`, `Humidity layer A` and `Humidity layer B`.
For each feature apply 3 main parameters of AutoWEKA: time limit, memory limit and parallel runs,
then metrics collected such as estimated error rate, training time on evaluation dataset, correlation coefficient,
mean absolute error, root mean squared error, relative absolute error, root relative squared error. For a quantitative characteristic of performance resources, we take the number of applied configurations, for efficiency, we take the ratio

![equation](https://latex.codecogs.com/svg.image?1%20-%20%5Cfrac%7BMAE_k%7D%7BMAE_0%7D)

where k is the run number.





