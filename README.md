# Automatic data frame processing for AutoML

Environment for knowledge analysis - WEKA:
https://github.com/Waikato/weka

AutoML - AutoWEKA:
https://github.com/automl/autoweka

Dataset - vertical farming:
https://www.kaggle.com/datasets/midouazerty/work-for-parmavir

## Iteration 1. Launching AutoWEKA, comparing the results.

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

\* CLI: '-timeLimit x -memLimit y -parallelRund z'

** metrics: correlation coefficient, mean absolute error, root mean squared error, reative squared error %, root relative squared error %

*** best classifier
