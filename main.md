# Modeling

In this section we propose a capsule in which three types of E.colis consuming ammonia, hydrogen sulfide and producing mycrene compete with each other and bacteria in the intestine. This is because they take up a similar niche with bacteria in the intestine environment. Consequently, some of our engineered E.colis may become extinct during competition after introduction, leading to the loss of the function of our product. 

## Preliminaries

We are inspired by the general logistic growth model, widely used to describe the population growth in biology, because it depicts the growth accurately in previously study. In our data, the growth curve is decided by the growth rate without inhibition and inhibition rate. The inhibition rate here refers to many inhibitive factors, such as the inhibition of the environment, the inhibition of population density, the inhibition of other bacteria... With the increase of the total population, the inhibition rate increase meanwhile, slowing down the marginal growth rate. When the inhibition rate is same as growth rate, meaning the marginal growth rate is equal to zero, the population reaches the highest capacity. 

Here are some basic assumptions of our model. First, assuming no other bacteria in the intestine, this model only considers competition among three engineered E.coli. Second, the growth rate of our E.coli in intestine is same as the data tested in vivo with lax anaerobic control due to technical limitation. Third, the environment of intestine--its pH value, temperature, humidity--is same to the LB medium we used in the experiment. Fourth, the growth rate of each E.coli would not be promoted or inhibited by the specific substances secreted by other bacteria or intestine itself. Fifth, the total population capacity of the bacteria in intestine keeps the same. Sixth, all our E.colis and nutrients distribute evenly in intestine and would not be excreted outside body during a certain period of time. 

Therefore, by looking out for the growth rate of each single E.coli, and their respective inhibition rate, we can then calculate and draw curves of concentrations rate and finally find the real-time ratio of each E.coli. Fortunately, with the best-fit logistic curve of population growth of each E.coli, we can calculate the growth rate and inhibition rate of each E.coli. 

Based on our experimental results, we trained a logistic regression model that predicts concentration  based on time. 

$$
\frac {dA} {dt} = kP(1-\frac{P}{N_m})
$$
taking the integral, we get
$$
A(t) = \frac{N_m}{1+\frac{N_m-P_0}{P_0} \cdot e^{-kt}} = \frac{N_m}{1+a \cdot e^{-kt}}
$$

Here, noted that $P_0$ can be trivial, we denote $\frac{N_m-P_0}{P_0}$ as an extra parameter $a$. 

In order to elongate the effective time as long as possible, we optimize our model with the following objectives,
* a simple logistic regression model, with parameters $a$, $k$ and $N_m$,  that maps a certain time to the corresponding population concentration that also takes into account the extinct ones. 
* display of the real-time ratio among three  three engineered E.coli that maximizes the minimum effective time among them.


```python
# from __future__ import muli
import itertools
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
plt.style.use("tableau-colorblind10")
import pandas as pd
from scipy import integrate
FS = (8, 4)
```


```python
# Some typical logistic curves
t = np.linspace(-5, 15, 1000)

fig = plt.figure(figsize=(10, 18))

ax = fig.add_subplot(3, 1, 1)
t0, L = 5., 10000.
for k in [0.5,1.,2.,4.]:
    D = L / (1. + np.exp(-k * (t - t0)))
    _ = plt.plot(t, D, label=f'k={k}')
_ = ax.legend()
_ = ax.set_xlabel('t')
```


    
![svg](main_files/main_2_0.svg)
    


## Data
Before training our model, we have to tidy up and pre-process our data. Here we use pandas.


```python
df = pd.read_csv('experimental_results.csv')
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
      <th>Cycle Nr.</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>59</th>
      <th>60</th>
      <th>61</th>
      <th>62</th>
      <th>63</th>
      <th>64</th>
      <th>65</th>
      <th>66</th>
      <th>67</th>
      <th>68</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Trial</td>
      <td>0</td>
      <td>637.9000</td>
      <td>1275.7000</td>
      <td>1913.5000</td>
      <td>2551.3000</td>
      <td>3189.4000</td>
      <td>3827.2000</td>
      <td>4465.1000</td>
      <td>5102.8000</td>
      <td>...</td>
      <td>36999.2000</td>
      <td>37637.1000</td>
      <td>38275.0000</td>
      <td>38912.9000</td>
      <td>39550.9000</td>
      <td>40188.8000</td>
      <td>40826.6000</td>
      <td>41464.5000</td>
      <td>42102.4000</td>
      <td>42740.3000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Temp. [°C]</td>
      <td>26.4</td>
      <td>37.3000</td>
      <td>37.4000</td>
      <td>37.5000</td>
      <td>37.2000</td>
      <td>37.1000</td>
      <td>37.5000</td>
      <td>37.6000</td>
      <td>37.2000</td>
      <td>...</td>
      <td>37.3000</td>
      <td>37.7000</td>
      <td>37.1000</td>
      <td>37.5000</td>
      <td>37.5000</td>
      <td>37.2000</td>
      <td>37.7000</td>
      <td>37.1000</td>
      <td>37.2000</td>
      <td>37.6000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A01</td>
      <td>0.045000002</td>
      <td>0.0449</td>
      <td>0.0452</td>
      <td>0.0451</td>
      <td>0.0459</td>
      <td>0.0463</td>
      <td>0.0476</td>
      <td>0.0485</td>
      <td>0.0498</td>
      <td>...</td>
      <td>0.8127</td>
      <td>0.8032</td>
      <td>0.8087</td>
      <td>0.7994</td>
      <td>0.8060</td>
      <td>0.7975</td>
      <td>0.7849</td>
      <td>0.7939</td>
      <td>0.7780</td>
      <td>0.7924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A02</td>
      <td>0.045899998</td>
      <td>0.0459</td>
      <td>0.0460</td>
      <td>0.0464</td>
      <td>0.0468</td>
      <td>0.0476</td>
      <td>0.0486</td>
      <td>0.0496</td>
      <td>0.0509</td>
      <td>...</td>
      <td>0.8858</td>
      <td>0.8895</td>
      <td>0.9066</td>
      <td>0.9023</td>
      <td>0.9309</td>
      <td>0.9392</td>
      <td>0.9300</td>
      <td>0.9581</td>
      <td>0.9345</td>
      <td>0.9616</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A03</td>
      <td>0.045600001</td>
      <td>0.0458</td>
      <td>0.0461</td>
      <td>0.0466</td>
      <td>0.0471</td>
      <td>0.0483</td>
      <td>0.0502</td>
      <td>0.0508</td>
      <td>0.0523</td>
      <td>...</td>
      <td>0.7782</td>
      <td>0.7730</td>
      <td>0.7863</td>
      <td>0.7877</td>
      <td>0.7953</td>
      <td>0.7901</td>
      <td>0.7775</td>
      <td>0.7842</td>
      <td>0.7684</td>
      <td>0.7794</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A04</td>
      <td>0.044599999</td>
      <td>0.0448</td>
      <td>0.0451</td>
      <td>0.0453</td>
      <td>0.0457</td>
      <td>0.0462</td>
      <td>0.0478</td>
      <td>0.0483</td>
      <td>0.0496</td>
      <td>...</td>
      <td>0.8160</td>
      <td>0.8141</td>
      <td>0.8176</td>
      <td>0.8104</td>
      <td>0.8172</td>
      <td>0.8085</td>
      <td>0.7979</td>
      <td>0.8039</td>
      <td>0.7900</td>
      <td>0.8030</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A05</td>
      <td>0.044500001</td>
      <td>0.0449</td>
      <td>0.0455</td>
      <td>0.0457</td>
      <td>0.0459</td>
      <td>0.0467</td>
      <td>0.0484</td>
      <td>0.0491</td>
      <td>0.0511</td>
      <td>...</td>
      <td>0.9163</td>
      <td>0.9056</td>
      <td>0.9344</td>
      <td>0.9267</td>
      <td>0.9338</td>
      <td>0.9434</td>
      <td>0.9260</td>
      <td>0.9494</td>
      <td>0.9215</td>
      <td>0.9427</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A06</td>
      <td>0.045699999</td>
      <td>0.0459</td>
      <td>0.0470</td>
      <td>0.0476</td>
      <td>0.0464</td>
      <td>0.0474</td>
      <td>0.0488</td>
      <td>0.0496</td>
      <td>0.0512</td>
      <td>...</td>
      <td>0.8082</td>
      <td>0.8017</td>
      <td>0.8100</td>
      <td>0.8059</td>
      <td>0.8210</td>
      <td>0.8239</td>
      <td>0.8162</td>
      <td>0.8273</td>
      <td>0.8120</td>
      <td>0.8228</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A07</td>
      <td>0.0471</td>
      <td>0.0481</td>
      <td>0.0481</td>
      <td>0.0480</td>
      <td>0.0482</td>
      <td>0.0485</td>
      <td>0.0496</td>
      <td>0.0501</td>
      <td>0.0506</td>
      <td>...</td>
      <td>0.4180</td>
      <td>0.4175</td>
      <td>0.4294</td>
      <td>0.4291</td>
      <td>0.4394</td>
      <td>0.4410</td>
      <td>0.4398</td>
      <td>0.4528</td>
      <td>0.4471</td>
      <td>0.4623</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A08</td>
      <td>0.0462</td>
      <td>0.0474</td>
      <td>0.0490</td>
      <td>0.0484</td>
      <td>0.0495</td>
      <td>0.0493</td>
      <td>0.0521</td>
      <td>0.0529</td>
      <td>0.0541</td>
      <td>...</td>
      <td>0.8316</td>
      <td>0.7980</td>
      <td>0.8750</td>
      <td>0.8514</td>
      <td>0.8851</td>
      <td>0.8929</td>
      <td>0.8690</td>
      <td>0.9004</td>
      <td>0.8711</td>
      <td>0.9012</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A09</td>
      <td>0.046300001</td>
      <td>0.0464</td>
      <td>0.0478</td>
      <td>0.0472</td>
      <td>0.0478</td>
      <td>0.0482</td>
      <td>0.0498</td>
      <td>0.0505</td>
      <td>0.0515</td>
      <td>...</td>
      <td>0.6156</td>
      <td>0.6130</td>
      <td>0.6210</td>
      <td>0.6258</td>
      <td>0.6271</td>
      <td>0.6297</td>
      <td>0.6172</td>
      <td>0.6345</td>
      <td>0.6247</td>
      <td>0.6464</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A10</td>
      <td>0.044500001</td>
      <td>0.0451</td>
      <td>0.0457</td>
      <td>0.0455</td>
      <td>0.0457</td>
      <td>0.0460</td>
      <td>0.0467</td>
      <td>0.0473</td>
      <td>0.0480</td>
      <td>...</td>
      <td>0.4182</td>
      <td>0.4271</td>
      <td>0.4322</td>
      <td>0.4338</td>
      <td>0.4472</td>
      <td>0.4527</td>
      <td>0.4484</td>
      <td>0.4640</td>
      <td>0.4548</td>
      <td>0.4717</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A11</td>
      <td>0.0449</td>
      <td>0.0456</td>
      <td>0.0463</td>
      <td>0.0467</td>
      <td>0.0465</td>
      <td>0.0478</td>
      <td>0.0486</td>
      <td>0.0507</td>
      <td>0.0548</td>
      <td>...</td>
      <td>0.6892</td>
      <td>0.6892</td>
      <td>0.7047</td>
      <td>0.6901</td>
      <td>0.6716</td>
      <td>0.7227</td>
      <td>0.6563</td>
      <td>0.7107</td>
      <td>0.6773</td>
      <td>0.7081</td>
    </tr>
    <tr>
      <th>13</th>
      <td>A12</td>
      <td>0.0528</td>
      <td>0.0539</td>
      <td>0.0545</td>
      <td>0.0536</td>
      <td>0.0550</td>
      <td>0.0588</td>
      <td>0.0566</td>
      <td>0.0569</td>
      <td>0.0569</td>
      <td>...</td>
      <td>0.5434</td>
      <td>0.5435</td>
      <td>0.5541</td>
      <td>0.5484</td>
      <td>0.5630</td>
      <td>0.5613</td>
      <td>0.5541</td>
      <td>0.5691</td>
      <td>0.5525</td>
      <td>0.5686</td>
    </tr>
    <tr>
      <th>14</th>
      <td>B01</td>
      <td>0.050500002</td>
      <td>0.0518</td>
      <td>0.0511</td>
      <td>0.0528</td>
      <td>0.0509</td>
      <td>0.0530</td>
      <td>0.0523</td>
      <td>0.0545</td>
      <td>0.0564</td>
      <td>...</td>
      <td>0.8370</td>
      <td>0.8284</td>
      <td>0.8333</td>
      <td>0.8256</td>
      <td>0.8292</td>
      <td>0.8236</td>
      <td>0.8124</td>
      <td>0.8208</td>
      <td>0.8072</td>
      <td>0.8184</td>
    </tr>
    <tr>
      <th>15</th>
      <td>B02</td>
      <td>0.046100002</td>
      <td>0.0468</td>
      <td>0.0464</td>
      <td>0.0474</td>
      <td>0.0476</td>
      <td>0.0483</td>
      <td>0.0494</td>
      <td>0.0504</td>
      <td>0.0523</td>
      <td>...</td>
      <td>0.9409</td>
      <td>0.9354</td>
      <td>0.9474</td>
      <td>0.9476</td>
      <td>0.9513</td>
      <td>0.9501</td>
      <td>0.9331</td>
      <td>0.9495</td>
      <td>0.9286</td>
      <td>0.9472</td>
    </tr>
    <tr>
      <th>16</th>
      <td>B03</td>
      <td>0.046</td>
      <td>0.0457</td>
      <td>0.0466</td>
      <td>0.0472</td>
      <td>0.0473</td>
      <td>0.0483</td>
      <td>0.0498</td>
      <td>0.0509</td>
      <td>0.0526</td>
      <td>...</td>
      <td>0.8154</td>
      <td>0.8152</td>
      <td>0.8286</td>
      <td>0.8279</td>
      <td>0.8331</td>
      <td>0.8310</td>
      <td>0.8171</td>
      <td>0.8191</td>
      <td>0.8069</td>
      <td>0.8159</td>
    </tr>
    <tr>
      <th>17</th>
      <td>B04</td>
      <td>0.046500001</td>
      <td>0.0470</td>
      <td>0.0469</td>
      <td>0.0473</td>
      <td>0.0475</td>
      <td>0.0480</td>
      <td>0.0491</td>
      <td>0.0504</td>
      <td>0.0512</td>
      <td>...</td>
      <td>0.8830</td>
      <td>0.8838</td>
      <td>0.8998</td>
      <td>0.9071</td>
      <td>0.9219</td>
      <td>0.9309</td>
      <td>0.9285</td>
      <td>0.9496</td>
      <td>0.9464</td>
      <td>0.9701</td>
    </tr>
    <tr>
      <th>18</th>
      <td>B05</td>
      <td>0.0473</td>
      <td>0.0480</td>
      <td>0.0487</td>
      <td>0.0489</td>
      <td>0.0491</td>
      <td>0.0494</td>
      <td>0.0511</td>
      <td>0.0526</td>
      <td>0.0538</td>
      <td>...</td>
      <td>0.8917</td>
      <td>0.8887</td>
      <td>0.9072</td>
      <td>0.9189</td>
      <td>0.9215</td>
      <td>0.9412</td>
      <td>0.9266</td>
      <td>0.9549</td>
      <td>0.9657</td>
      <td>0.9871</td>
    </tr>
    <tr>
      <th>19</th>
      <td>B06</td>
      <td>0.045600001</td>
      <td>0.0459</td>
      <td>0.0460</td>
      <td>0.0465</td>
      <td>0.0467</td>
      <td>0.0476</td>
      <td>0.0491</td>
      <td>0.0503</td>
      <td>0.0515</td>
      <td>...</td>
      <td>0.8962</td>
      <td>0.8838</td>
      <td>0.8892</td>
      <td>0.8845</td>
      <td>0.8885</td>
      <td>0.8863</td>
      <td>0.8810</td>
      <td>0.8865</td>
      <td>0.8761</td>
      <td>0.8840</td>
    </tr>
    <tr>
      <th>20</th>
      <td>B07</td>
      <td>0.202000007</td>
      <td>0.2207</td>
      <td>0.2312</td>
      <td>0.0507</td>
      <td>0.0489</td>
      <td>0.0504</td>
      <td>0.0500</td>
      <td>0.0510</td>
      <td>0.0531</td>
      <td>...</td>
      <td>0.4398</td>
      <td>0.4510</td>
      <td>0.4430</td>
      <td>0.4473</td>
      <td>0.4525</td>
      <td>0.4538</td>
      <td>0.4736</td>
      <td>0.4658</td>
      <td>0.4620</td>
      <td>0.4732</td>
    </tr>
    <tr>
      <th>21</th>
      <td>B08</td>
      <td>0.0504</td>
      <td>0.0495</td>
      <td>0.0502</td>
      <td>0.0504</td>
      <td>0.0510</td>
      <td>0.0517</td>
      <td>0.0530</td>
      <td>0.0539</td>
      <td>0.0558</td>
      <td>...</td>
      <td>0.8915</td>
      <td>0.8987</td>
      <td>0.9028</td>
      <td>0.9319</td>
      <td>0.8974</td>
      <td>0.9322</td>
      <td>0.8824</td>
      <td>0.9245</td>
      <td>0.9195</td>
      <td>0.9259</td>
    </tr>
    <tr>
      <th>22</th>
      <td>B09</td>
      <td>0.0473</td>
      <td>0.0472</td>
      <td>0.0480</td>
      <td>0.0480</td>
      <td>0.0488</td>
      <td>0.0491</td>
      <td>0.0517</td>
      <td>0.0512</td>
      <td>0.0522</td>
      <td>...</td>
      <td>0.5819</td>
      <td>0.5810</td>
      <td>0.5936</td>
      <td>0.5963</td>
      <td>0.6031</td>
      <td>0.6012</td>
      <td>0.5948</td>
      <td>0.6107</td>
      <td>0.6014</td>
      <td>0.6110</td>
    </tr>
    <tr>
      <th>23</th>
      <td>B10</td>
      <td>0.054299999</td>
      <td>0.0544</td>
      <td>0.0535</td>
      <td>0.0549</td>
      <td>0.0541</td>
      <td>0.0556</td>
      <td>0.0549</td>
      <td>0.0564</td>
      <td>0.0571</td>
      <td>...</td>
      <td>0.7874</td>
      <td>0.8405</td>
      <td>0.8204</td>
      <td>0.8416</td>
      <td>0.9155</td>
      <td>0.9514</td>
      <td>0.9047</td>
      <td>1.0304</td>
      <td>0.9612</td>
      <td>1.0241</td>
    </tr>
    <tr>
      <th>24</th>
      <td>B11</td>
      <td>0.0469</td>
      <td>0.0474</td>
      <td>0.0472</td>
      <td>0.0473</td>
      <td>0.0474</td>
      <td>0.0480</td>
      <td>0.0492</td>
      <td>0.0502</td>
      <td>0.0517</td>
      <td>...</td>
      <td>0.7282</td>
      <td>0.7716</td>
      <td>0.7649</td>
      <td>0.7431</td>
      <td>0.7847</td>
      <td>0.8052</td>
      <td>0.7814</td>
      <td>0.7784</td>
      <td>0.7643</td>
      <td>0.7782</td>
    </tr>
    <tr>
      <th>25</th>
      <td>B12</td>
      <td>0.045400001</td>
      <td>0.0457</td>
      <td>0.0460</td>
      <td>0.0464</td>
      <td>0.0463</td>
      <td>0.0468</td>
      <td>0.0481</td>
      <td>0.0486</td>
      <td>0.0498</td>
      <td>...</td>
      <td>0.7579</td>
      <td>0.7459</td>
      <td>0.7710</td>
      <td>0.7711</td>
      <td>0.7909</td>
      <td>0.7966</td>
      <td>0.7713</td>
      <td>0.8015</td>
      <td>0.7855</td>
      <td>0.8102</td>
    </tr>
    <tr>
      <th>26</th>
      <td>C01</td>
      <td>0.047499999</td>
      <td>0.0477</td>
      <td>0.0488</td>
      <td>0.0487</td>
      <td>0.0484</td>
      <td>0.0494</td>
      <td>0.0507</td>
      <td>0.0516</td>
      <td>0.0530</td>
      <td>...</td>
      <td>0.8661</td>
      <td>0.8822</td>
      <td>0.8618</td>
      <td>0.8550</td>
      <td>0.8585</td>
      <td>0.8528</td>
      <td>0.8436</td>
      <td>0.8493</td>
      <td>0.8381</td>
      <td>0.8471</td>
    </tr>
    <tr>
      <th>27</th>
      <td>C02</td>
      <td>0.051100001</td>
      <td>0.0516</td>
      <td>0.0503</td>
      <td>0.0515</td>
      <td>0.0509</td>
      <td>0.0551</td>
      <td>0.0576</td>
      <td>0.0546</td>
      <td>0.0571</td>
      <td>...</td>
      <td>0.9389</td>
      <td>0.9238</td>
      <td>0.9419</td>
      <td>0.9389</td>
      <td>0.9407</td>
      <td>0.9355</td>
      <td>0.9312</td>
      <td>0.9236</td>
      <td>0.9053</td>
      <td>0.8985</td>
    </tr>
    <tr>
      <th>28</th>
      <td>C03</td>
      <td>0.047899999</td>
      <td>0.0477</td>
      <td>0.0493</td>
      <td>0.0484</td>
      <td>0.0494</td>
      <td>0.0496</td>
      <td>0.0521</td>
      <td>0.0530</td>
      <td>0.0539</td>
      <td>...</td>
      <td>0.8159</td>
      <td>0.8142</td>
      <td>0.8180</td>
      <td>0.8111</td>
      <td>0.8131</td>
      <td>0.8049</td>
      <td>0.7944</td>
      <td>0.7976</td>
      <td>0.7843</td>
      <td>0.7924</td>
    </tr>
    <tr>
      <th>29</th>
      <td>C04</td>
      <td>0.046</td>
      <td>0.0454</td>
      <td>0.0457</td>
      <td>0.0460</td>
      <td>0.0462</td>
      <td>0.0468</td>
      <td>0.0478</td>
      <td>0.0486</td>
      <td>0.0501</td>
      <td>...</td>
      <td>0.8595</td>
      <td>0.8445</td>
      <td>0.8480</td>
      <td>0.8435</td>
      <td>0.8456</td>
      <td>0.8411</td>
      <td>0.8317</td>
      <td>0.8367</td>
      <td>0.8251</td>
      <td>0.8335</td>
    </tr>
    <tr>
      <th>30</th>
      <td>C05</td>
      <td>0.045600001</td>
      <td>0.0454</td>
      <td>0.0458</td>
      <td>0.0468</td>
      <td>0.0464</td>
      <td>0.0471</td>
      <td>0.0484</td>
      <td>0.0502</td>
      <td>0.0515</td>
      <td>...</td>
      <td>0.9158</td>
      <td>0.9278</td>
      <td>0.9401</td>
      <td>0.9536</td>
      <td>0.9846</td>
      <td>1.0126</td>
      <td>1.0248</td>
      <td>1.0672</td>
      <td>1.0725</td>
      <td>1.1112</td>
    </tr>
    <tr>
      <th>31</th>
      <td>C06</td>
      <td>0.046300001</td>
      <td>0.0466</td>
      <td>0.0466</td>
      <td>0.0474</td>
      <td>0.0471</td>
      <td>0.0480</td>
      <td>0.0491</td>
      <td>0.0506</td>
      <td>0.0521</td>
      <td>...</td>
      <td>0.8684</td>
      <td>0.8677</td>
      <td>0.8728</td>
      <td>0.8677</td>
      <td>0.8702</td>
      <td>0.8632</td>
      <td>0.8534</td>
      <td>0.8585</td>
      <td>0.8466</td>
      <td>0.8556</td>
    </tr>
    <tr>
      <th>32</th>
      <td>C07</td>
      <td>0.046599999</td>
      <td>0.0499</td>
      <td>0.0483</td>
      <td>0.0490</td>
      <td>0.0526</td>
      <td>0.0484</td>
      <td>0.0487</td>
      <td>0.0496</td>
      <td>0.0565</td>
      <td>...</td>
      <td>0.4189</td>
      <td>0.4193</td>
      <td>0.4291</td>
      <td>0.4299</td>
      <td>0.4440</td>
      <td>0.4430</td>
      <td>0.4421</td>
      <td>0.4555</td>
      <td>0.4502</td>
      <td>0.4633</td>
    </tr>
    <tr>
      <th>33</th>
      <td>C08</td>
      <td>0.046999998</td>
      <td>0.0471</td>
      <td>0.0476</td>
      <td>0.0481</td>
      <td>0.0483</td>
      <td>0.0488</td>
      <td>0.0503</td>
      <td>0.0512</td>
      <td>0.0535</td>
      <td>...</td>
      <td>0.8910</td>
      <td>0.9199</td>
      <td>0.9165</td>
      <td>0.9207</td>
      <td>0.9447</td>
      <td>0.9188</td>
      <td>0.9286</td>
      <td>0.9362</td>
      <td>0.9502</td>
      <td>0.9435</td>
    </tr>
    <tr>
      <th>34</th>
      <td>C09</td>
      <td>0.0515</td>
      <td>0.0517</td>
      <td>0.0503</td>
      <td>0.0539</td>
      <td>0.0522</td>
      <td>0.0563</td>
      <td>0.0537</td>
      <td>0.0567</td>
      <td>0.0576</td>
      <td>...</td>
      <td>0.6343</td>
      <td>0.6424</td>
      <td>0.6439</td>
      <td>0.6449</td>
      <td>0.6652</td>
      <td>0.6560</td>
      <td>0.6598</td>
      <td>0.6582</td>
      <td>0.6595</td>
      <td>0.6723</td>
    </tr>
    <tr>
      <th>35</th>
      <td>C10</td>
      <td>0.046799999</td>
      <td>0.0472</td>
      <td>0.0473</td>
      <td>0.0480</td>
      <td>0.0474</td>
      <td>0.0482</td>
      <td>0.0489</td>
      <td>0.0493</td>
      <td>0.0497</td>
      <td>...</td>
      <td>0.7022</td>
      <td>0.7031</td>
      <td>0.7131</td>
      <td>0.7146</td>
      <td>0.7292</td>
      <td>0.7294</td>
      <td>0.7273</td>
      <td>0.7427</td>
      <td>0.7418</td>
      <td>0.7595</td>
    </tr>
    <tr>
      <th>36</th>
      <td>C11</td>
      <td>0.046999998</td>
      <td>0.0473</td>
      <td>0.0466</td>
      <td>0.0476</td>
      <td>0.0470</td>
      <td>0.0488</td>
      <td>0.0489</td>
      <td>0.0506</td>
      <td>0.0537</td>
      <td>...</td>
      <td>0.8090</td>
      <td>0.7906</td>
      <td>0.8176</td>
      <td>0.8181</td>
      <td>0.8283</td>
      <td>0.8325</td>
      <td>0.8038</td>
      <td>0.8421</td>
      <td>0.8173</td>
      <td>0.8465</td>
    </tr>
    <tr>
      <th>37</th>
      <td>C12</td>
      <td>0.046</td>
      <td>0.0463</td>
      <td>0.0461</td>
      <td>0.0467</td>
      <td>0.0464</td>
      <td>0.0472</td>
      <td>0.0482</td>
      <td>0.0481</td>
      <td>0.0492</td>
      <td>...</td>
      <td>0.6923</td>
      <td>0.6816</td>
      <td>0.7194</td>
      <td>0.7252</td>
      <td>0.7478</td>
      <td>0.7448</td>
      <td>0.7334</td>
      <td>0.7608</td>
      <td>0.7418</td>
      <td>0.7736</td>
    </tr>
    <tr>
      <th>38</th>
      <td>D01</td>
      <td>0.043900002</td>
      <td>0.0437</td>
      <td>0.0443</td>
      <td>0.0447</td>
      <td>0.0437</td>
      <td>0.0435</td>
      <td>0.0441</td>
      <td>0.0435</td>
      <td>0.0433</td>
      <td>...</td>
      <td>0.0430</td>
      <td>0.0430</td>
      <td>0.0426</td>
      <td>0.0429</td>
      <td>0.0429</td>
      <td>0.0427</td>
      <td>0.0426</td>
      <td>0.0424</td>
      <td>0.0426</td>
      <td>0.0426</td>
    </tr>
    <tr>
      <th>39</th>
      <td>D02</td>
      <td>0.045000002</td>
      <td>0.0454</td>
      <td>0.0479</td>
      <td>0.0466</td>
      <td>0.0451</td>
      <td>0.0451</td>
      <td>0.0448</td>
      <td>0.0449</td>
      <td>0.0446</td>
      <td>...</td>
      <td>0.7929</td>
      <td>0.8020</td>
      <td>0.8159</td>
      <td>0.8231</td>
      <td>0.8355</td>
      <td>0.8415</td>
      <td>0.8451</td>
      <td>0.8598</td>
      <td>0.8572</td>
      <td>0.8672</td>
    </tr>
    <tr>
      <th>40</th>
      <td>D03</td>
      <td>0.049199998</td>
      <td>0.0494</td>
      <td>0.0490</td>
      <td>0.0500</td>
      <td>0.0487</td>
      <td>0.0488</td>
      <td>0.0484</td>
      <td>0.0489</td>
      <td>0.0491</td>
      <td>...</td>
      <td>0.0486</td>
      <td>0.0484</td>
      <td>0.0488</td>
      <td>0.0488</td>
      <td>0.0489</td>
      <td>0.0489</td>
      <td>0.0489</td>
      <td>0.0488</td>
      <td>0.0491</td>
      <td>0.0490</td>
    </tr>
    <tr>
      <th>41</th>
      <td>D04</td>
      <td>0.044100001</td>
      <td>0.0440</td>
      <td>0.0441</td>
      <td>0.0448</td>
      <td>0.0437</td>
      <td>0.0440</td>
      <td>0.0441</td>
      <td>0.0439</td>
      <td>0.0437</td>
      <td>...</td>
      <td>0.5269</td>
      <td>0.5427</td>
      <td>0.5447</td>
      <td>0.5351</td>
      <td>0.5624</td>
      <td>0.5467</td>
      <td>0.5430</td>
      <td>0.5557</td>
      <td>0.5252</td>
      <td>0.5620</td>
    </tr>
    <tr>
      <th>42</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>43</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>44</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>45</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>46</th>
      <td>End Time:</td>
      <td>9/20/20 8:21</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>47 rows × 69 columns</p>
</div>



From the raw data, we observe that there exists 68 columns, for 68 recording period, and 39 groups of data, including 3 for the control groups. We notice that temperatures and cycle number should not taken account as features, nor should the control groups should be our training sets.

### Data Cleaning and EDA

To clean our data, we use `pandas` and list operations to remove some rows and indices. After cleaning, we end up with 36 groups of data. In every group of data, we have $68$ features, namely $68$ points of time, each labeled with a corresponding concentration rate. Since for every E.coli in every scenarios, there exist three groups of data and each points of time should be mapped to only one output, namely one concentration rate, we manually pair our groups in three for each scenarios, and calculate the respective average value as our training sets. 


```python
df.columns = df.loc[0].values
df.drop(df.index[0:2],inplace=True)
df.drop(df.index[36:],inplace=True)
total = df.groupby('Trial').sum().max(axis=1)
all_sets = total.index.to_list()
all_data =  pd.DataFrame()
for set in all_sets:
    temp = df[df['Trial'] == set][df.columns[1:]].T.sum(axis=1)
#     temp.index = pd.to_datetime(temp.index)
    temp = temp.to_frame(set)
    all_data = pd.concat([all_data, temp], axis=1)
all_data['$argA^fbr$ <antibio>'] = all_data[['A01', 'B01','C01','A07', 'B07','C07']].mean(axis=1)
all_data['$cysE-mut$ <antibio>'] = all_data[['A02', 'B02','C02','A08', 'B08','C08']].mean(axis=1)
all_data['$myrcene$ <antibio>'] = all_data[['A03', 'B03','C03','A09', 'B09','C09']].mean(axis=1)
all_data['$argA^fbr$'] = all_data[['A04', 'B04','C04','A10', 'B10','C10']].mean(axis=1)
all_data['$cysE-mut$'] = all_data[['A05', 'B05','C05','A11', 'B11','C11']].mean(axis=1)
all_data['$myrcene$'] = all_data[['A06', 'B06','C06','A12', 'B12','C12']].mean(axis=1)
# all_data['avg11_antibio'] = all_data[['A07', 'B07','C07']].mean(axis=1)
# all_data['avg22_antibio'] = all_data[['A08', 'B08','C08']].mean(axis=1)
# all_data['avg33_antibio'] = all_data[[].mean(axis=1)
# all_data['avg44_antibio'] = all_data[[].mean(axis=1)
# all_data['avg55_antibio'] = all_data[[].mean(axis=1)
# all_data['avg66_antibio'] = all_data[[]].mean(axis=1)
```

Now our clean data is visualized as followed,


```python
all_data
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
      <th>A01</th>
      <th>A02</th>
      <th>A03</th>
      <th>A04</th>
      <th>A05</th>
      <th>A06</th>
      <th>A07</th>
      <th>A08</th>
      <th>A09</th>
      <th>A10</th>
      <th>...</th>
      <th>C09</th>
      <th>C10</th>
      <th>C11</th>
      <th>C12</th>
      <th>$argA^fbr$ &lt;antibio&gt;</th>
      <th>$cysE-mut$ &lt;antibio&gt;</th>
      <th>$myrcene$ &lt;antibio&gt;</th>
      <th>$argA^fbr$</th>
      <th>$cysE-mut$</th>
      <th>$myrcene$</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0450</td>
      <td>0.0459</td>
      <td>0.0456</td>
      <td>0.0446</td>
      <td>0.0445</td>
      <td>0.0457</td>
      <td>0.0471</td>
      <td>0.0462</td>
      <td>0.0463</td>
      <td>0.0445</td>
      <td>...</td>
      <td>0.0515</td>
      <td>0.0468</td>
      <td>0.0470</td>
      <td>0.0460</td>
      <td>0.073117</td>
      <td>0.047783</td>
      <td>0.047433</td>
      <td>0.047117</td>
      <td>0.046033</td>
      <td>0.046967</td>
    </tr>
    <tr>
      <th>637.9</th>
      <td>0.0449</td>
      <td>0.0459</td>
      <td>0.0458</td>
      <td>0.0448</td>
      <td>0.0449</td>
      <td>0.0459</td>
      <td>0.0481</td>
      <td>0.0474</td>
      <td>0.0464</td>
      <td>0.0451</td>
      <td>...</td>
      <td>0.0517</td>
      <td>0.0472</td>
      <td>0.0473</td>
      <td>0.0463</td>
      <td>0.077183</td>
      <td>0.048050</td>
      <td>0.047417</td>
      <td>0.047317</td>
      <td>0.046433</td>
      <td>0.047383</td>
    </tr>
    <tr>
      <th>1275.7</th>
      <td>0.0452</td>
      <td>0.0460</td>
      <td>0.0461</td>
      <td>0.0451</td>
      <td>0.0455</td>
      <td>0.0470</td>
      <td>0.0481</td>
      <td>0.0490</td>
      <td>0.0478</td>
      <td>0.0457</td>
      <td>...</td>
      <td>0.0503</td>
      <td>0.0473</td>
      <td>0.0466</td>
      <td>0.0461</td>
      <td>0.078783</td>
      <td>0.048250</td>
      <td>0.048017</td>
      <td>0.047367</td>
      <td>0.046683</td>
      <td>0.047700</td>
    </tr>
    <tr>
      <th>1913.5</th>
      <td>0.0451</td>
      <td>0.0464</td>
      <td>0.0466</td>
      <td>0.0453</td>
      <td>0.0457</td>
      <td>0.0476</td>
      <td>0.0480</td>
      <td>0.0484</td>
      <td>0.0472</td>
      <td>0.0455</td>
      <td>...</td>
      <td>0.0539</td>
      <td>0.0480</td>
      <td>0.0476</td>
      <td>0.0467</td>
      <td>0.049050</td>
      <td>0.048700</td>
      <td>0.048550</td>
      <td>0.047833</td>
      <td>0.047167</td>
      <td>0.048033</td>
    </tr>
    <tr>
      <th>2551.3</th>
      <td>0.0459</td>
      <td>0.0468</td>
      <td>0.0471</td>
      <td>0.0457</td>
      <td>0.0459</td>
      <td>0.0464</td>
      <td>0.0482</td>
      <td>0.0495</td>
      <td>0.0478</td>
      <td>0.0457</td>
      <td>...</td>
      <td>0.0522</td>
      <td>0.0474</td>
      <td>0.0470</td>
      <td>0.0464</td>
      <td>0.049150</td>
      <td>0.049017</td>
      <td>0.048767</td>
      <td>0.047767</td>
      <td>0.047050</td>
      <td>0.047983</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>40188.8</th>
      <td>0.7975</td>
      <td>0.9392</td>
      <td>0.7901</td>
      <td>0.8085</td>
      <td>0.9434</td>
      <td>0.8239</td>
      <td>0.4410</td>
      <td>0.8929</td>
      <td>0.6297</td>
      <td>0.4527</td>
      <td>...</td>
      <td>0.6560</td>
      <td>0.7294</td>
      <td>0.8325</td>
      <td>0.7448</td>
      <td>0.635283</td>
      <td>0.928117</td>
      <td>0.718817</td>
      <td>0.785667</td>
      <td>0.876267</td>
      <td>0.779350</td>
    </tr>
    <tr>
      <th>40826.6</th>
      <td>0.7849</td>
      <td>0.9300</td>
      <td>0.7775</td>
      <td>0.7979</td>
      <td>0.9260</td>
      <td>0.8162</td>
      <td>0.4398</td>
      <td>0.8690</td>
      <td>0.6172</td>
      <td>0.4484</td>
      <td>...</td>
      <td>0.6598</td>
      <td>0.7273</td>
      <td>0.8038</td>
      <td>0.7334</td>
      <td>0.632733</td>
      <td>0.912383</td>
      <td>0.710133</td>
      <td>0.773083</td>
      <td>0.853150</td>
      <td>0.768233</td>
    </tr>
    <tr>
      <th>41464.5</th>
      <td>0.7939</td>
      <td>0.9581</td>
      <td>0.7842</td>
      <td>0.8039</td>
      <td>0.9494</td>
      <td>0.8273</td>
      <td>0.4528</td>
      <td>0.9004</td>
      <td>0.6345</td>
      <td>0.4640</td>
      <td>...</td>
      <td>0.6582</td>
      <td>0.7427</td>
      <td>0.8421</td>
      <td>0.7608</td>
      <td>0.639683</td>
      <td>0.932050</td>
      <td>0.717383</td>
      <td>0.804550</td>
      <td>0.883783</td>
      <td>0.783950</td>
    </tr>
    <tr>
      <th>42102.4</th>
      <td>0.7780</td>
      <td>0.9345</td>
      <td>0.7684</td>
      <td>0.7900</td>
      <td>0.9215</td>
      <td>0.8120</td>
      <td>0.4471</td>
      <td>0.8711</td>
      <td>0.6247</td>
      <td>0.4548</td>
      <td>...</td>
      <td>0.6595</td>
      <td>0.7418</td>
      <td>0.8173</td>
      <td>0.7418</td>
      <td>0.630433</td>
      <td>0.918200</td>
      <td>0.707533</td>
      <td>0.786550</td>
      <td>0.869767</td>
      <td>0.769083</td>
    </tr>
    <tr>
      <th>42740.3</th>
      <td>0.7924</td>
      <td>0.9616</td>
      <td>0.7794</td>
      <td>0.8030</td>
      <td>0.9427</td>
      <td>0.8228</td>
      <td>0.4623</td>
      <td>0.9012</td>
      <td>0.6464</td>
      <td>0.4717</td>
      <td>...</td>
      <td>0.6723</td>
      <td>0.7595</td>
      <td>0.8465</td>
      <td>0.7736</td>
      <td>0.642783</td>
      <td>0.929650</td>
      <td>0.719567</td>
      <td>0.810317</td>
      <td>0.895633</td>
      <td>0.785800</td>
    </tr>
  </tbody>
</table>
<p>68 rows × 42 columns</p>
</div>



We can also visualize some of our experimental as follows. After connecting all the data points, we see that, in general, the curves are in _S_ shape.


```python
start = 0
ax = all_data[['A01', 'B01','C01','A07','B07','C07','$argA^fbr$ <antibio>']][start:].plot(style='-', figsize=(15,9))
markers = itertools.cycle(("o", "v", "^", "<", ">", "s", "p", "P", "*", "h", "X", "D", '.'))
for i, line in enumerate(ax.get_lines()):
    marker = next(markers)
    line.set_marker(marker)
_ = ax.legend()
```


    
![svg](main_files/main_10_0.svg)
    


## Nonlinear Least Squares for $argA^{fbr}$ with Antibacterial

When looking at the data, we only have the concentration rates per time period. We also have the formula that we want to apply, but we do not yet have the correct values of the parameters $a$, $k$ and $N_m$ in the formula.

Unfortunately, it is not possible to rewrite the Logistic Function as a Linear Regression, as was the case for the Exponential model. We will therefore need a more complex method: Nonlinear Least Squares estimation.

**Define the logistic function that has to be fitted.** First, we define the logistic function with input point of time $t$ and parameters $a$, $k$, $N_m$ and an offset to accommodate our model with the non-zero concentration rate at the begining.


```python
def logistic(t, a, k, N_m, offset):
    return N_m / (1 + a * np.exp(-k*t)) + offset
```

**Random Initialization of parameters and upper, lower bounds set up.** Next, we use `np.random.random` to initialize our parameters, set up a relatively high bounds to let the model free. 


```python
p0 = np.random.random(size=4)
bounds = (0., [100176.,3.,10019834.,10000.])
```

**Use SciPy's Curve Fit for Nonlinear Least Squares Estimation.**
In this step, Scipy does a Nonlinear Least Squares optimization, which minimizes the following lost function $\ell$,

$$
\ell(a,k,N_m,\textrm{offset}) = \sum_{i=0}^T \|r_i\|^2,
$$

where a residual \|r_i\|, the error distance matrix between ground-truth label concentraton rate and the predicted one, is given by,

$$
r_i = y^{true}_i - f(a,k,N_m,\textrm{offset}) 
$$
Here, $f$ is the logistic model to train.


```python
import scipy.optimize as optim
x = np.array([float(x) for x in all_data.index])
y = np.array([float(x) for x in all_data['$argA^fbr$ <antibio>']])
(a,k,N_m,offset),cov = optim.curve_fit(logistic, x, y, bounds=bounds, p0=p0)
a,k,N_m,offset
```




    (81.26818615403548,
     0.00022831205475778584,
     0.5963894442011484,
     0.03403331139255463)



**Plot the fitted function vs the real data.** As shown in the graph below, our Logistic model is very close to the ground truths.


```python
test_logistic = lambda t : N_m / (1 + a * np.exp(-k*t)) + offset
plt.scatter(x, y)
plt.plot(x, test_logistic(x))
plt.title('Logistic Models v. Real Observation of group $argA^fbr$ <antibio>')
plt.title('Logistic Models v. Real Observation of group ')
plt.legend(['Logistic Model','Experimental Data'])
plt.xlabel('Time/(s)')
plt.ylabel('Concentration')
```




    Text(0, 0.5, 'Concentration')




    
![svg](main_files/main_18_1.svg)
    


## Nonlinear Least Squares for all cases

In this section, we apply the same training algorithm, as we did to $argA^{fbr}$ with antibacterial, to all $6$ scenrios. We also visualize all the fitted curves alongside our original data. We observe that none of the curves are obviously under-fitted.


```python
ret_data = {
    'group_name':[],
    'a':[],
    'k':[],
    'N_m':[],
    'offset':[],
    }
for group in all_data.columns[all_data.columns.get_loc('$argA^fbr$ <antibio>'):]:
    p0 = np.random.random(size=4)
    y = np.array([float(x) for x in all_data[group]])
    (a,k,N_m,offset),cov = optim.curve_fit(logistic, x, y, bounds=bounds, p0=p0)
    test_logistic = lambda t : N_m / (1 + a * np.exp(-k*t)) + offset
    plt.scatter(x, y)
    ret_data['group_name'].append(group)
    ret_data['a'].append(a)
    ret_data['k'].append(k)
    ret_data['N_m'].append(N_m)
    ret_data['offset'].append(offset)
    plt.plot(x, test_logistic(x),marker = next(markers))
    plt.title(f'Logistic Models v. Real Observation of all groups')
    plt.xlabel('Time/(s)')
    plt.ylabel('Concentration')
```


    
![svg](main_files/main_20_0.svg)
    


We also display all the trained parameters in a `pandas.DataFrame`.


```python
output_df = pd.DataFrame(ret_data)
output_df.to_csv('./outputs/out.csv')
output_df
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
      <th>group_name</th>
      <th>a</th>
      <th>k</th>
      <th>N_m</th>
      <th>offset</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$argA^fbr$ &lt;antibio&gt;</td>
      <td>81.268446</td>
      <td>0.000228</td>
      <td>0.596389</td>
      <td>3.403340e-02</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$cysE-mut$ &lt;antibio&gt;</td>
      <td>24.231418</td>
      <td>0.000189</td>
      <td>0.897522</td>
      <td>2.719961e-26</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$myrcene$ &lt;antibio&gt;</td>
      <td>60.918889</td>
      <td>0.000218</td>
      <td>0.693091</td>
      <td>2.001465e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$argA^fbr$</td>
      <td>56.760447</td>
      <td>0.000191</td>
      <td>0.771270</td>
      <td>1.323637e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>$cysE-mut$</td>
      <td>35.515316</td>
      <td>0.000261</td>
      <td>0.805412</td>
      <td>1.094864e-25</td>
    </tr>
    <tr>
      <th>5</th>
      <td>$myrcene$</td>
      <td>48.169640</td>
      <td>0.000201</td>
      <td>0.761660</td>
      <td>9.799030e-03</td>
    </tr>
  </tbody>
</table>
</div>



## Applying the Parameters into Three E.colis Coexisting Scenarios

Given the formula of derivatives of concentrations of the three crafted e.colis,

$$
\frac {dA}{dt} = k_A \cdot A (1 - \frac {A}{N_1} - \frac{k_B}{k_A} \cdot \frac{B}{N_2} - \frac{k_C}{k_A} \cdot \frac{C}{N_3}) = k_A \cdot A (1 - \frac {A}{N_1}) - \frac{k_B}{N_2} \cdot AB - \frac{k_C}{N_3} \cdot AC \\
\frac {dB}{dt} = k_B \cdot B (1 - \frac {B}{N_2} - \frac{k_A}{k_B} \cdot \frac{A}{N_1} - \frac{k_C}{k_B} \cdot \frac{C}{N_3}) = k_B \cdot B (1 - \frac {B}{N_2}) - \frac{k_A}{N_1} \cdot BA - \frac{k_C}{N_3} \cdot BC \\
\frac {dC}{dt} = k_C \cdot C (1 - \frac {C}{N_3} - \frac{k_A}{k_C} \cdot \frac{A}{N_1} - \frac{k_B}{k_C} \cdot \frac{B}{N_2}) = k_C \cdot C (1 - \frac {C}{N_3}) - \frac{k_A}{N_1} \cdot AC - \frac{k_B}{N_2} \cdot BC
$$

we plot the curve of growth rate and concentrations of three crafted
e.coli with and without the effect of antibacterial.


```python
from scipy.integrate import odeint
from scipy.optimize import leastsq
```


```python
t = np.arange(0,80000,1000)

def deriv(w,t,ka,kb,kc,n1,n2,n3): 
    x,y,z = w
    return np.array([ka*(1-x/n1)*x-kb*y*x/n2-kc*x*z/n3,
                     kb*(1-y/n2)*y-ka*y*x/n1-kc*y*z/n3,
                     kc*(1-z/n3)*z-ka*z*x/n1-kb*y*z/n2])

fig = plt.figure(figsize=(10, 18))
plt.subplots_adjust(hspace=0.2)

ax1 = fig.add_subplot(3,1,1)
p = ret_data['k'][:3] + ret_data['N_m'][:3] + [0.02,0.02,0.02]
ka,kb,kc,n1,n2,n3,x0,y0,z0=p
yinit = np.array([x0,y0,z0]) # initial vals
yyy = odeint(deriv,yinit,t,args=(ka,kb,kc,n1,n2,n3))
ax1.plot(t,yyy[:,0],marker = next(markers),label="$argA^{fbr}$")
ax1.plot(t,yyy[:,1],marker = next(markers),label="$cysE-mut$")
ax1.plot(t,yyy[:,2],marker = next(markers),label="$myrcene$")
ax1.set_xlabel('time/(s)')
ax1.set_ylabel('concentration')
ax1.set_title('Three engineer e.colis\' real-time concentrations comparison antibacterial')
_ = ax1.legend(loc=2)


ax2 = fig.add_subplot(3,1,2)

p = ret_data['k'][3:] + ret_data['N_m'][3:] + [0.02,0.02,0.02]

ka,kb,kc,n1,n2,n3,x0,y0,z0=p
yinit = np.array([x0,y0,z0]) # initial vals
yyy = odeint(deriv,yinit,t,args=(ka,kb,kc,n1,n2,n3))
ax2.plot(t,yyy[:,0],marker = next(markers),label="$argA^{fbr}$")
ax2.plot(t,yyy[:,1],marker = next(markers),label="$cysE-mut$")
ax2.plot(t,yyy[:,2],marker = next(markers),label="$myrcene$")
ax2.set_xlabel('time/(s)')
ax2.set_ylabel('concentration')
ax2.set_title('Three engineer e.colis\' real-time concentrations comparison no antibacterial')
_ = ax2.legend(loc=2)
```


    
![svg](main_files/main_25_0.svg)
    



```python
t = np.arange(0,500000,5000)

def deriv(w,t,ka,kb,kc,n1,n2,n3,offset1=0,offset2=0):
    x,y,z = w
    y+=offset1
    z+=offset2
    return np.array([ka*(1-x/n1)*x-kb*y*x/n2-kc*x*z/n3,
                     kb*(1-y/n2)*y-ka*y*x/n1-kc*y*z/n3,
                     kc*(1-z/n3)*z-ka*z*x/n1-kb*y*z/n2])

fig = plt.figure(figsize=(10, 18))
plt.subplots_adjust(hspace=0.2)

ax1 = fig.add_subplot(3,1,1)
p = ret_data['k'][:3] + ret_data['N_m'][:3] + [0.02,0.02,0.02]
ka,kb,kc,n1,n2,n3,x0,y0,z0=p
yinit = np.array([x0,y0,z0]) # initial vals
yyy1 = odeint(deriv,yinit,t,args=(ka,kb,kc,n1,n2,n3))
ax1.plot(t,yyy1[:,0],marker = next(markers),label="$argA^{fbr}$")
ax1.plot(t,yyy1[:,1],marker = next(markers),label="$cysE-mut$")
ax1.plot(t,yyy1[:,2],marker = next(markers),label="$myrcene$")
ax1.plot([0,500000],[0.6,0.6],"g--")
ax1.plot([0,500000],[0,0],"g--")
ax1.text(400000,0.6,'y = 0.6')
ax1.text(400000,0,'y = 0\n')
ax1.set_xlabel('time/(s)')
ax1.set_ylabel('concentration')
ax1.set_title('Three engineer e.colis\' real-time concentrations comparison antibacterial')
_ = ax1.legend(loc=2)


ax2 = fig.add_subplot(3,1,2)

p = ret_data['k'][3:] + ret_data['N_m'][3:] + [0.02,0.02,0.02]

ka,kb,kc,n1,n2,n3,x0,y0,z0=p
yinit = np.array([x0,y0,z0]) # initial vals
yyy2 = odeint(deriv,yinit,t,args=(ka,kb,kc,n1,n2,n3))
ax2.plot(t,yyy2[:,0],marker = next(markers),label="$argA^{fbr}$")
ax2.plot(t,yyy2[:,1],marker = next(markers),label="$cysE-mut$")
ax2.plot(t,yyy2[:,2],marker = next(markers),label="$myrcene$")
ax2.plot([0,500000],[0.8,0.8],"g--")
ax2.plot([0,500000],[0,0],"g--")
ax2.text(400000,0.8,'y = 0.8')
ax2.text(400000,0,'y = 0\n')
ax2.set_xlabel('time/(s)')
ax2.set_ylabel('concentration')
ax2.set_title('Three engineer e.colis\' real-time concentrations comparison no antibacterial')
_ = ax2.legend(loc=2)
```


    
![svg](main_files/main_26_0.svg)
    



```python
from 
```


```python
yyy1
```




    array([[2.00000000e-02, 2.00000000e-02, 2.00000000e-02],
           [5.38249907e-02, 4.43143562e-02, 5.11295740e-02],
           [1.19040813e-01, 8.06894407e-02, 1.07416819e-01],
           [1.98989292e-01, 1.11047993e-01, 1.70566702e-01],
           [2.60464413e-01, 1.19671240e-01, 2.12080692e-01],
           [2.97077531e-01, 1.12375490e-01, 2.29779214e-01],
           [3.19249526e-01, 9.94242627e-02, 2.34562927e-01],
           [3.35017023e-01, 8.58992434e-02, 2.33821353e-01],
           [3.48011122e-01, 7.34642306e-02, 2.30727097e-01],
           [3.59629101e-01, 6.25025854e-02, 2.26489717e-01],
           [3.70391210e-01, 5.29985715e-02, 2.21586101e-01],
           [3.80501585e-01, 4.48250000e-02, 2.16235252e-01],
           [3.90053995e-01, 3.78310948e-02, 2.10563427e-01],
           [3.99103753e-01, 3.18691394e-02, 2.04659640e-01],
           [4.07692078e-01, 2.68026130e-02, 1.98594332e-01],
           [4.15854056e-01, 2.25084786e-02, 1.92425978e-01],
           [4.23621139e-01, 1.88774424e-02, 1.86203812e-01],
           [4.31021891e-01, 1.58133973e-02, 1.79969311e-01],
           [4.38082227e-01, 1.32324921e-02, 1.73757257e-01],
           [4.44825554e-01, 1.10620556e-02, 1.67596614e-01],
           [4.51272919e-01, 9.23943382e-03, 1.61511325e-01],
           [4.57443169e-01, 7.71086703e-03, 1.55521002e-01],
           [4.63353153e-01, 6.43039806e-03, 1.49641557e-01],
           [4.69017910e-01, 5.35892258e-03, 1.43885712e-01],
           [4.74450903e-01, 4.46312552e-03, 1.38263552e-01],
           [4.79664146e-01, 3.71488816e-03, 1.32782813e-01],
           [4.84668458e-01, 3.09039286e-03, 1.27449327e-01],
           [4.89473594e-01, 2.56955869e-03, 1.22267277e-01],
           [4.94088358e-01, 2.13547441e-03, 1.17239450e-01],
           [4.98520828e-01, 1.77391757e-03, 1.12367480e-01],
           [5.02778387e-01, 1.47294660e-03, 1.07652004e-01],
           [5.06867847e-01, 1.22255048e-03, 1.03092824e-01],
           [5.10795552e-01, 1.01433509e-03, 9.86890513e-02],
           [5.14567444e-01, 8.41269729e-04, 9.44392132e-02],
           [5.18189112e-01, 6.97489842e-04, 9.03413411e-02],
           [5.21665856e-01, 5.78098330e-04, 8.63930583e-02],
           [5.25002731e-01, 4.78993789e-04, 8.25916580e-02],
           [5.28204574e-01, 3.96759230e-04, 7.89341564e-02],
           [5.31276038e-01, 3.28549903e-04, 7.54173450e-02],
           [5.34221610e-01, 2.72013036e-04, 7.20378280e-02],
           [5.37045643e-01, 2.25139090e-04, 6.87920929e-02],
           [5.39752343e-01, 1.86292335e-04, 6.56765065e-02],
           [5.42345820e-01, 1.54111651e-04, 6.26873612e-02],
           [5.44830055e-01, 1.27462536e-04, 5.98208983e-02],
           [5.47208935e-01, 1.05399319e-04, 5.70733331e-02],
           [5.49486245e-01, 8.71377324e-05, 5.44408710e-02],
           [5.51665677e-01, 7.20261698e-05, 5.19197256e-02],
           [5.53750828e-01, 5.95242224e-05, 4.95061324e-02],
           [5.55745206e-01, 4.91818697e-05, 4.71963627e-02],
           [5.57652230e-01, 4.06281355e-05, 4.49867306e-02],
           [5.59475225e-01, 3.35567509e-05, 4.28736080e-02],
           [5.61217427e-01, 2.77136933e-05, 4.08534289e-02],
           [5.62881988e-01, 2.28847017e-05, 3.89226951e-02],
           [5.64471972e-01, 1.88940913e-05, 3.70779848e-02],
           [5.65990351e-01, 1.55969441e-05, 3.53159551e-02],
           [5.67440012e-01, 1.28726918e-05, 3.36333465e-02],
           [5.68823754e-01, 1.06224073e-05, 3.20269848e-02],
           [5.70144285e-01, 8.76623747e-06, 3.04937823e-02],
           [5.71404229e-01, 7.23346393e-06, 2.90307439e-02],
           [5.72606091e-01, 5.96837544e-06, 2.76349612e-02],
           [5.73752366e-01, 4.92365536e-06, 2.63036228e-02],
           [5.74845409e-01, 4.06139236e-06, 2.50340047e-02],
           [5.75887517e-01, 3.34984195e-06, 2.38234754e-02],
           [5.76880883e-01, 2.76271364e-06, 2.26694921e-02],
           [5.77827636e-01, 2.27830231e-06, 2.15696026e-02],
           [5.78729818e-01, 1.87875407e-06, 2.05214450e-02],
           [5.79589399e-01, 1.54921973e-06, 1.95227421e-02],
           [5.80408275e-01, 1.27738626e-06, 1.85712999e-02],
           [5.81188266e-01, 1.05317914e-06, 1.76650100e-02],
           [5.81931123e-01, 8.68263244e-07, 1.68018448e-02],
           [5.82638531e-01, 7.15646276e-07, 1.59798522e-02],
           [5.83312099e-01, 5.89695115e-07, 1.51971659e-02],
           [5.83953372e-01, 4.85896136e-07, 1.44519966e-02],
           [5.84563830e-01, 4.00414273e-07, 1.37426249e-02],
           [5.85144892e-01, 3.30079142e-07, 1.30674047e-02],
           [5.85697920e-01, 2.72074878e-07, 1.24247534e-02],
           [5.86224213e-01, 2.24201800e-07, 1.18131577e-02],
           [5.86725022e-01, 1.84709906e-07, 1.12311657e-02],
           [5.87201528e-01, 1.52200023e-07, 1.06774057e-02],
           [5.87654903e-01, 1.25414368e-07, 1.01505395e-02],
           [5.88086235e-01, 1.03330949e-07, 9.64930403e-03],
           [5.88496539e-01, 8.51309356e-08, 9.17248273e-03],
           [5.88886835e-01, 7.01304789e-08, 8.71891610e-03],
           [5.89258032e-01, 5.77922320e-08, 8.28754123e-03],
           [5.89611094e-01, 4.76104399e-08, 7.87723913e-03],
           [5.89946867e-01, 3.92213717e-08, 7.48702964e-03],
           [5.90266178e-01, 3.23113533e-08, 7.11595008e-03],
           [5.90569819e-01, 2.66158257e-08, 6.76307888e-03],
           [5.90858544e-01, 2.19209031e-08, 6.42754155e-03],
           [5.91133072e-01, 1.80516952e-08, 6.10850210e-03],
           [5.91394095e-01, 1.48628183e-08, 5.80516268e-03],
           [5.91642265e-01, 1.22410695e-08, 5.51676715e-03],
           [5.91878179e-01, 1.00821319e-08, 5.24259098e-03],
           [5.92102462e-01, 8.30382723e-09, 4.98194233e-03],
           [5.92315670e-01, 6.83905472e-09, 4.73416335e-03],
           [5.92518346e-01, 5.63243448e-09, 4.49862536e-03],
           [5.92711001e-01, 4.63868991e-09, 4.27473347e-03],
           [5.92894109e-01, 3.82173309e-09, 4.06193572e-03],
           [5.93068163e-01, 3.14782466e-09, 3.85965959e-03],
           [5.93233597e-01, 2.59241544e-09, 3.66740215e-03]])




```python
THRESHOLD = 1
t = np.arange(0,500000,1)

def add_offset(params,threshold=THRESHOLD):
    o1,o2 = params
    def deriv(w,t,ka,kb,kc,n1,n2,n3,offset1=0,offset2=0):
        x,y,z = w
        y+=offset1
        z+=offset2
        return np.array([ka*(1-x/n1)*x-kb*y*x/n2-kc*x*z/n3,
                     kb*(1-y/n2)*y-ka*y*x/n1-kc*y*z/n3,
                     kc*(1-z/n3)*z-ka*z*x/n1-kb*y*z/n2])
    p = ret_data['k'][3:] + ret_data['N_m'][3:] + [0.02,0.02,0.02]
    ka,kb,kc,n1,n2,n3,x0,y0,z0=p
    yinit = np.array([x0,y0,z0])
    yyy = odeint(deriv,yinit,t,args=(ka,kb,kc,n1,n2,n3,o1,o2))
    for i in range(1,yyy.shape[0]):
        j = min(yyy[i])
        if j <= 0:
             return t[i]
             break
```


```python
add_offset((0.,0))
```




    326326




```python
def maximize(f,init_args):
    l = 1e-5
    x,y = init_args
    for i in range(10,100000):
        l*=0.9
        arr = np.array([[x+l,y+l],[x+l,y-l],[x+l,y],[x,y+l],[x,y-l],[x,y],[x-l,y+l],[x-l,y-l],[x-l,y]])
        f_group = map(f,arr)
        x,y = arr[np.argmax(f)]
    return x,y

res = maximize(add_offset,(0,0))
add_offset(res)
```




    150956




```python
optimize.newton(f, x0, args=(y,))
```




    124206


