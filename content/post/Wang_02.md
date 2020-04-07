---
title: Assignment 02 KNN
date: 2020-04-02

# Put any other Academic metadata here...
---


[download](https://raw.githubusercontent.com/lxw8502/academic-kickstart/master/static/files/Wang_02.ipynb "download the ipynb file!")

```python
import pandas as pd
import numpy as np
```


## loading dataset


```python
datafile_path = './iris.data'
title = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_dataset = pd.read_csv(datafile_path,names=title)
iris_dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



## a. Random divide the dataset as development and test. 80% as develop data and 20% as test data


```python
# Hierarchical division
# class1_data = iris_dataset.iloc[:50]
# class2_data = iris_dataset.iloc[50:100]
# class3_data = iris_dataset.iloc[100:150]

# develop_data = pd.concat( [class1_data[:40],class2_data[:40],class3_data[:40]], axis=0 )
# test_data = pd.concat( [class1_data[40:50],class2_data[40:50],class3_data[40:50]], axis=0 )
# develop_data

# Random division
boolean_list = np.random.rand(len(iris_dataset)) < 0.8
develop_data = iris_dataset[boolean_list]
test_data  = iris_dataset[~boolean_list]
print(len(develop_data),len(test_data))
```

    118 32
    


```python
develop_data
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.7</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>142</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>144</th>
      <td>6.7</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>2.5</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>145</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
<p>118 rows × 5 columns</p>
</div>




```python
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>8</th>
      <td>4.4</td>
      <td>2.9</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>9</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5.8</td>
      <td>4.0</td>
      <td>1.2</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.7</td>
      <td>4.4</td>
      <td>1.5</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>16</th>
      <td>5.4</td>
      <td>3.9</td>
      <td>1.3</td>
      <td>0.4</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>22</th>
      <td>4.6</td>
      <td>3.6</td>
      <td>1.0</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>30</th>
      <td>4.8</td>
      <td>3.1</td>
      <td>1.6</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>35</th>
      <td>5.0</td>
      <td>3.2</td>
      <td>1.2</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>37</th>
      <td>4.9</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.1</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>43</th>
      <td>5.0</td>
      <td>3.5</td>
      <td>1.6</td>
      <td>0.6</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>48</th>
      <td>5.3</td>
      <td>3.7</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>50</th>
      <td>7.0</td>
      <td>3.2</td>
      <td>4.7</td>
      <td>1.4</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>55</th>
      <td>5.7</td>
      <td>2.8</td>
      <td>4.5</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>65</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>4.4</td>
      <td>1.4</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>67</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>4.1</td>
      <td>1.0</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>68</th>
      <td>6.2</td>
      <td>2.2</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>77</th>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>1.7</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>81</th>
      <td>5.5</td>
      <td>2.4</td>
      <td>3.7</td>
      <td>1.0</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>84</th>
      <td>5.4</td>
      <td>3.0</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>87</th>
      <td>6.3</td>
      <td>2.3</td>
      <td>4.4</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>89</th>
      <td>5.5</td>
      <td>2.5</td>
      <td>4.0</td>
      <td>1.3</td>
      <td>Iris-versicolor</td>
    </tr>
    <tr>
      <th>101</th>
      <td>5.8</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>102</th>
      <td>7.1</td>
      <td>3.0</td>
      <td>5.9</td>
      <td>2.1</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>120</th>
      <td>6.9</td>
      <td>3.2</td>
      <td>5.7</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>123</th>
      <td>6.3</td>
      <td>2.7</td>
      <td>4.9</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>124</th>
      <td>6.7</td>
      <td>3.3</td>
      <td>5.7</td>
      <td>2.1</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>125</th>
      <td>7.2</td>
      <td>3.2</td>
      <td>6.0</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>137</th>
      <td>6.4</td>
      <td>3.1</td>
      <td>5.5</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>143</th>
      <td>6.8</td>
      <td>3.2</td>
      <td>5.9</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>



## b.Implement kNN


```python
from math import pow, sqrt

def euclidean_distance(data1, data2):
    distance = []
    for i in range(len(data1) - 1):
        value = pow(data1[i] - data2[i], 2)
        distance.append(value)
    return sqrt(sum(distance))

euclidean_distance(test_data.iloc[0],test_data.iloc[1])
```




    0.5099019513592785




```python
def normalized_euclidean_distance(data1, data2):
    data1_sum = 0
    data2_sum = 0
    for i in range(len(data1) - 1):
        data1_sum += data1[i]
        data2_sum += data2[i]
    
    data1_mean = data1_sum/len(data1)
    data2_mean = data2_sum/len(data2)
    
    numerator_square = 0
    data1_square = 0
    data2_square = 0
    for i in range(len(data1) - 1):
        numerator_square += pow((data1[i] - data1_mean) - (data2[i] - data2_mean),2)
        data1_square += pow(data1[i] - data1_mean, 2)
        data2_square += pow(data2[i] - data2_mean, 2)
    return 0.5 * sqrt(numerator_square) / (sqrt(data1_square)+ sqrt(data2_square))

normalized_euclidean_distance(test_data.iloc[0],test_data.iloc[1])
```




    0.030044451366651233




```python
def cosine_similarity(data1, data2):
    numerator = 0
    data1_square = 0
    data2_square = 0
    for i in range(len(data1) - 1):
        numerator += data1[i] * data2[i]
        data1_square += pow(data1[i], 2 )
        data2_square += pow(data2[i], 2 )
    return numerator/ (sqrt(data1_square) * sqrt(data2_square))

cosine_similarity(test_data.iloc[0],test_data.iloc[1])
```




    0.99926089097567




```python
def get_neighbors(dataset, target, number, distance_method):
    distance_list = list()
    if distance_method == 'euclidean_distance':
        method = euclidean_distance
    elif distance_method == 'normalized_euclidean_distance':
        method = normalized_euclidean_distance
    else:
        method = cosine_similarity
    for index, dataset_item in dataset.iterrows():
        distance = method(target, dataset_item)
        distance_list.append((distance, dataset_item))
    
    distance_list.sort(key=lambda tup: tup[0])
    if(distance_method == 'cosine_similarity'):
        distance_list.reverse()
    neighbors_list = list()
    for i in range(number):
        neighbors_list.append(distance_list[i][1])
    return neighbors_list

get_neighbors(develop_data, test_data.iloc[0], 5, 'cosine_similarity')
```




    [sepal_length            4.8
     sepal_width               3
     petal_length            1.4
     petal_width             0.1
     class           Iris-setosa
     Name: 12, dtype: object,
     sepal_length            4.8
     sepal_width               3
     petal_length            1.4
     petal_width             0.3
     class           Iris-setosa
     Name: 45, dtype: object,
     sepal_length            5.4
     sepal_width             3.4
     petal_length            1.7
     petal_width             0.2
     class           Iris-setosa
     Name: 20, dtype: object,
     sepal_length            4.9
     sepal_width             3.1
     petal_length            1.5
     petal_width             0.1
     class           Iris-setosa
     Name: 34, dtype: object,
     sepal_length              5
     sepal_width               3
     petal_length            1.6
     petal_width             0.2
     class           Iris-setosa
     Name: 25, dtype: object]




```python
def predict(dataset, target, number, distance_method):
    neighbors_list = get_neighbors(dataset, target, number, distance_method)
    output_values = [row.values[-1] for row in neighbors_list]
    prediction_class = max(set(output_values), key=output_values.count)
    return prediction_class

predict(develop_data, test_data.iloc[0], 5, 'cosine_similarity')
```




    'Iris-setosa'



## c.Using the development dataset, find optimal hyperparameters


```python
def calculate_accuracy(dataset, develop_dataset, number, distance_method):
    predictions = []
    count = 0
    for index, dataset_item in dataset.iterrows():
        label = predict(develop_dataset, dataset_item, number, distance_method)
        predictions.append(label)
    for i in range(len(predictions)):
        if dataset.iloc[i][-1] == predictions[i]:
            count += 1
    return count/len(predictions)
calculate_accuracy(develop_data, develop_data, 3, 'cosine_similarity')
```




    0.9830508474576272




```python
numbers = [1, 3 ,5, 7]
distance_methods = ['euclidean_distance', 'normalized_euclidean_distance', 'cosine_similarity']

predict_result = []
for method in distance_methods:
    inner =[]
    for number in numbers:
        accuracy = calculate_accuracy(develop_data, develop_data, number, method)
        inner.append(accuracy)
    predict_result.append(inner)
predict_result
```




    [[1.0, 0.9491525423728814, 0.9661016949152542, 0.9661016949152542],
     [1.0, 0.9830508474576272, 0.9745762711864406, 0.9830508474576272],
     [1.0, 0.9830508474576272, 0.9745762711864406, 0.9745762711864406]]



### Draw bar charts for accuracy


```python
import matplotlib.pyplot as plt

x = np.arange(len(numbers))  
width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, predict_result[0], width, label='euclidean distance')
rects2 = ax.bar(x, predict_result[1], width, label='normalized euclidean distance')
rects3 = ax.bar(x + width, predict_result[2], width, label='cosine similarity')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Neighbor number')
ax.set_xticks(x)
ax.set_xticklabels(numbers)
ax.legend()

fig.tight_layout()
plt.show()
```


![png](https://raw.githubusercontent.com/lxw8502/academic-kickstart/master/content/post/Wang_02_18_0.png)


## d.Using the test dataset

In the step c, we can discover when neighbor number is 1, the accuracy is largest. So I choose neighbor number is 1 and cosine similarity as the optimal hyperparameters.


```python
print('final accuracy:')
calculate_accuracy(test_data, develop_data, 1, 'cosine_similarity')
```

    final accuracy:0.96875


