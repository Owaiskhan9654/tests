# Que1 (a)


```python
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```

    [name: "/device:CPU:0"
    device_type: "CPU"
    memory_limit: 268435456
    locality {
    }
    incarnation: 8061388388049468646
    , name: "/device:XLA_CPU:0"
    device_type: "XLA_CPU"
    memory_limit: 17179869184
    locality {
    }
    incarnation: 17671265477961798381
    physical_device_desc: "device: XLA_CPU device"
    , name: "/device:GPU:0"
    device_type: "GPU"
    memory_limit: 4973462816
    locality {
      bus_id: 1
      links {
      }
    }
    incarnation: 11606881419384641507
    physical_device_desc: "device: 0, name: GeForce GTX 1660 Ti, pci bus id: 0000:01:00.0, compute capability: 7.5"
    , name: "/device:XLA_GPU:0"
    device_type: "XLA_GPU"
    memory_limit: 17179869184
    locality {
    }
    incarnation: 12858335633744657994
    physical_device_desc: "device: XLA_GPU device"
    ]
    


```python
import math
import pandas as pd
```


```python
x=[[0,3,0],[2,0,0],[0,1,3],[0,1,2],[-1,0,1],[1,1,1]]
```


```python
x
```




    [[0, 3, 0], [2, 0, 0], [0, 1, 3], [0, 1, 2], [-1, 0, 1], [1, 1, 1]]




```python
df=pd.DataFrame(x,columns=["X1","X2","X3"])
```


```python
df1=pd.DataFrame(['RED','RED','RED','GREEN','GREEN','RED'],columns=["COLOR"])
```


```python
df=pd.concat([df,df1], axis=1)
```


```python
df
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
      <th>X1</th>
      <th>X2</th>
      <th>X3</th>
      <th>COLOR</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>RED</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>RED</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>RED</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>GREEN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>GREEN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>RED</td>
    </tr>
  </tbody>
</table>
</div>




```python
test_point=[0,0,0]
```


```python
d1=math.sqrt((x[0][0]-test_point[0])**2+(x[0][1]-test_point[1])**2+(x[0][2]-test_point[2])**2) #Eucludian distance
```


```python
d1
```




    3.0




```python
d2=math.sqrt((x[1][0]-test_point[0])**2+(x[1][1]-test_point[1])**2+(x[1][2]-test_point[2])**2)
```


```python
d2
```




    2.0




```python
d3=math.sqrt((x[2][0]-test_point[0])**2+(x[2][1]-test_point[1])**2+(x[2][2]-test_point[2])**2)
```


```python
d3
```




    3.1622776601683795




```python
d4=math.sqrt((x[3][0]-test_point[0])**2+(x[3][1]-test_point[1])**2+(x[3][2]-test_point[2])**2)
```


```python
d4
```




    2.23606797749979




```python
d5=math.sqrt((x[4][0]-test_point[0])**2+(x[4][1]-test_point[1])**2+(x[4][2]-test_point[2])**2)
```


```python
d5
```




    1.4142135623730951




```python
d6=math.sqrt((x[5][0]-test_point[0])**2+(x[5][1]-test_point[1])**2+(x[5][2]-test_point[2])**2)
```


```python
d6
```




    1.7320508075688772



# Que 1(b) 

Prediction for K=1


```python
smallest_dist1=min(d1,d2,d3,d4,d5,d6)
```


```python
smallest_dist1
```




    1.4142135623730951



### Observation 5th has smallest distance which is of GREEN color so our test point will be considered as GREEN

 for k=1,our prediction for it to be green is 1 and prediction for it to be red is 0

# Que 1(c)

Prediction for k = 3

First we have to find 3 closest distance to the test point


```python
smallest_dist1=min(d1,d2,d3,d4,d5,d6)
```

### Observation 5th has smallest distance to test point which is of green color


```python
smallest_dist2=min(d1,d2,d3,d4,d6)#since observation 5th gives smallest distance so removed from the list
```


```python
smallest_dist2
```




    1.7320508075688772




```python
so our prediction
```


      File "<ipython-input-26-2263e5611722>", line 1
        so our predictiom
           ^
    SyntaxError: invalid syntax
    


### Observation 6th has second smallest distance which is of red color


```python
smallest_dist3=min(d1,d2,d3,d4) #since observation 5,6th gives two smallest distance so removed from the list
```


```python
smallest_dist3
```




    2.0



### Observation 2nd has third smallest distance which is also of red color

so  we have test point close to 1 GREEN observation and 2 RED observation ,So our prediction for it to be green is 1/3 and prediction for it to be red is 2/3

# Que 1(d)


```python

```
