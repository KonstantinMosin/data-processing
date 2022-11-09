# Automatic data frame processing for AutoML

Environment for knowledge analysis - WEKA:
https://github.com/Waikato/weka

AutoML - AutoWEKA:
https://github.com/automl/autoweka

Dataset - vertical farming:
https://www.kaggle.com/datasets/midouazerty/work-for-parmavir

## Iteration 1. Launching AutoWEKA, comparing the results.

| feature             | launch parameters*    | estimated error  | metrics*                                 | total of configuration | result*            |
|:------------------- | --------------------- |:----------------:| ---------------------------------------- |:----------------------:|:------------------:|
| Temperature layer A | 5 min, 1024MB, 1 run  | 0.56024657       | 0.9781, 0.3842, 0.5157, 18.5953, 21.1105 | 18                     | trees.RandomForest |
| Temperature layer A | 15 min, 2048MB, 4 run | 0.56024657       | 0.9781, 0.3842, 0.5157, 18.5953, 21.1105 | 89                     | trees.RandomForest |
| Temperature layer B | 5 min, 1024MB, 1 run  | 0.57685781       | 0.9778, 0.3794, 0.5131, 18.2875, 21.1949 | 23                     | trees.RandomForest |
| Temperature layer B | 15 min, 2048MB, 4 run | 0.57685781       | 0.9778, 0.3794, 0.5131, 18.2875, 21.1949 | 23                     | trees.RandomForest |
| Humidity layer A    | 5 min, 1024MB, 1 run  | 0.98145939       | 0.3704, 0.7756, 0.9815, 92.7262, 92.8883 | 58                     | trees.M5P          |
| Humidity layer A    | 15 min, 2048MB, 4 run | 0.93240144       | 0.4704, 0.7382, 0.9324, 88.2446, 88.2453 | 177                    | trees.RandomTree   |
| Humidity layer B    | 5 min, 1024MB, 1 run  | 0.97352648       | 0.4358, 0.7695, 0.9735, 89.5807, 90.1227 | 49                     | trees.M5P          |
| Humidity layer B    | 15 min, 2048MB, 4 run | 0.94186592       | 0.4897, 0.7469, 0.9419, 86.9592, 87.1918 | 182                    | trees.M5P          |
