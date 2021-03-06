# NeoCov
> Semantic change and socio-semantic variation. The case of Covid-related neologisms on Reddit.


## Description


This repository contains the code for the paper _Semantic change and socio-semantic variation. The case of Covid-related neologisms on Reddit_. This paper has been submitted and is currently under (anonymous) review for the journal _Linguistics Vanguard_.

You can clone the repository and install the code as a Python package named `neocov` by running `pip install .` within the cloned directory. This will automatically install all dependencies. As always, it is recommended to install this package in a virtual environment (e.g. using `conda`). 

The Reddit data used for this paper are too big to make them available here. Some parts of the code cannot be executed without having access to these datasets. The full datasets of Reddit comments and the models trained from these comments can be requested via email once the anonymous review process is finished. The datasets and models allow to reproduce our results.

This repository provides the code used to process the Reddit comments, train the models, and produce the results presented in our paper. The code was written and documented using the literate programming framework `nbdev` and the documentation is available here:

https://wuqui.github.io/neocov/

The code used for the tables and figures contained in the paper can be found directly via the following links:

| Reference | Link                                            |
|-----------|-------------------------------------------------|
| Table 2   | [semantic neologisms](https://wuqui.github.io/neocov/#semantic-neologisms)     |
| Figure 1  | [Covid-related communities](https://wuqui.github.io/neocov/#covid-communities) |
| Figure 2  | [Semantic axes](https://wuqui.github.io/neocov/#sem-axis)                      |
| Figure 3  | [Semantic maps for _vaccines_](https://wuqui.github.io/neocov/#sem-maps)       |



## Imports 

```python
# all_data
```

```python
%load_ext autoreload
%autoreload 2
```

```python
from neocov.read_data import *
from neocov.preproc import *
from neocov.type_emb import *
from neocov.communities import *
```

```python
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 100)
import altair as alt
from altair_saver import save
from gensim.models import Word2Vec
import pickle
```

## Variables

```python
DATA_DIR = '../data/'
COMMENTS_DIAC_DIR = f'{DATA_DIR}comments/by_date/'
COMMENTS_DIR_SUBR = f'{DATA_DIR}comments/subr/'
OUT_DIR = '../out/'
```

## Detecting semantic change

#### Read comments

```python
YEAR = '2020'
```

```python
comment_paths_year = get_comments_paths_year(COMMENTS_DIAC_DIR, YEAR)
```

```python
comments = read_comm_csvs(comment_paths_year)
```

```python
comments
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
      <th>author</th>
      <th>body</th>
      <th>created_utc</th>
      <th>id</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Broncos57</td>
      <td>Oh okay thank you so much for the reply! I rea...</td>
      <td>2020-04-14 21:20:57</td>
      <td>fnf0nqd</td>
      <td>boston</td>
    </tr>
    <tr>
      <th>1</th>
      <td>tresclow</td>
      <td>Es tan deprimente ver cuando esta clase de est...</td>
      <td>2020-04-14 21:20:57</td>
      <td>fnf0noq</td>
      <td>chile</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hicklebear</td>
      <td>This comment is Codex approved.</td>
      <td>2020-04-14 21:20:57</td>
      <td>fnf0nor</td>
      <td>Grimdank</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[deleted]</td>
      <td>[removed]</td>
      <td>2020-04-14 21:20:57</td>
      <td>fnf0nos</td>
      <td>acturnips</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ilovedog5</td>
      <td>Am I the only person who thinks this whole thi...</td>
      <td>2020-04-14 21:20:57</td>
      <td>fnf0not</td>
      <td>UnresolvedMysteries</td>
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
      <th>9599965</th>
      <td>Driedrain</td>
      <td>Invoice sent!</td>
      <td>2020-08-19 21:59:59</td>
      <td>g25e6cy</td>
      <td>hardwareswap</td>
    </tr>
    <tr>
      <th>9599966</th>
      <td>PresOfTheLesbianClub</td>
      <td>Yes. Fixed that. Thank you!</td>
      <td>2020-08-19 21:59:59</td>
      <td>g25e6cz</td>
      <td>vanderpumprules</td>
    </tr>
    <tr>
      <th>9599967</th>
      <td>originalasteele</td>
      <td>This is incredible!! Oh, how I miss Midna</td>
      <td>2020-08-19 21:59:59</td>
      <td>g25e6d1</td>
      <td>zelda</td>
    </tr>
    <tr>
      <th>9599968</th>
      <td>sunbeam2z</td>
      <td>I boosted you. I don't need a boost.</td>
      <td>2020-08-19 21:59:59</td>
      <td>g25e6ci</td>
      <td>Earnin</td>
    </tr>
    <tr>
      <th>9599969</th>
      <td>Unicornius</td>
      <td>Yeah its weird, I think Jamal has passed to hi...</td>
      <td>2020-08-19 21:59:59</td>
      <td>g25e6bf</td>
      <td>denvernuggets</td>
    </tr>
  </tbody>
</table>
<p>9599970 rows ?? 5 columns</p>
</div>



### Preprocessing

```python
comments_clean = clean_comments(comments)
```

    conv_to_lowerc       (9599970, 5) 0:00:08.393617      
    rm_punct             (9599970, 5) 0:00:57.095767      
    tokenize             (9599970, 5) 0:04:38.659268      
    rem_short_comments   (5125011, 5) 0:01:27.863816      


Dataset of comments after pre-processing:

```python
comments_clean.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 5125011 entries, 0 to 9599969
    Data columns (total 5 columns):
     #   Column       Dtype         
    ---  ------       -----         
     0   author       string        
     1   body         object        
     2   created_utc  datetime64[ns]
     3   id           string        
     4   subreddit    string        
    dtypes: datetime64[ns](1), object(1), string(3)
    memory usage: 234.6+ MB


```python
docs = comments_clean['body'].to_list()
```

Saving the cleaned comments to disk:

```python
with open(f'{OUT_DIR}docs_clean/diac_{YEAR}.pickle', 'wb') as fp:
    pickle.dump(docs, fp)
```

Loading the cleaned comments from disk:

```python
with open(f'{OUT_DIR}docs_clean/diac_{YEAR}.pickle', 'rb') as fp:
    docs = pickle.load(fp)
```

### Train models

#### Create corpus

```python
corpus = Corpus(docs)
```

#### Train model

```python
%%time
model = train_model(corpus, EPOCHS=20)
```

```python
len(model.wv.key_to_index)
```

#### Save model

```python
model.save(f'{OUT_DIR}models/{YEAR}_ep-20.model')
```

### Load models

```python
model_2019 = Word2Vec.load(f'{OUT_DIR}models/2019.model')
```

```python
model_2020 = Word2Vec.load(f'{OUT_DIR}models/2020.model')
```

### Align models

```python
model_2019_vocab = len(model_2019.wv.key_to_index)
model_2020_vocab = len(model_2020.wv.key_to_index)
```

```python
smart_procrustes_align_gensim(model_2019, model_2020)
```

    190756 190756
    190756 190756





    <gensim.models.word2vec.Word2Vec at 0x1738f1030>



```python
assert len(model_2019.wv.key_to_index) == len(model_2020.wv.vectors)
```

Overview of vocabulary sizes for both models before Procrustes alignment:

```python
models_vocab = pd.DataFrame(
    columns=['Model', 'Words'],
    data=[
        ['2019', model_2019_vocab],
        ['2020', model_2020_vocab],
        ['intersection', len(model_2019.wv.key_to_index)]
    ],
)

models_vocab
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
      <th>Model</th>
      <th>Words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>252564</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>277707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>intersection</td>
      <td>190756</td>
    </tr>
  </tbody>
</table>
</div>



```python
models_vocab.to_csv(f'{OUT_DIR}models_vocab.csv', index=False)
```

### Measure distances

Measuring semantic distances (~ cosine distance) between the 2019 and the 2020 model for all words contained in the aligned vocabulary.

```python
distances = measure_distances(model_2019, model_2020)
```

<a id='semantic-neologisms'></a>

20 words that show the highest semantic distance between 2019 and 2020. This output is presented in Table 2 in the paper.

```python
blacklist_lex = load_blacklist_lex()

k = 20
freq_min = 100

sem_change_cands = (distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')
    .query('lex.str.isalpha() == True')
    .query('lex.str.len() > 3')
    .query('lex not in @blacklist_lex')
    .nlargest(k, 'dist_sem')
    .reset_index(drop=True)
)

sem_change_cands
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
      <th>lex</th>
      <th>dist_sem</th>
      <th>freq_1</th>
      <th>freq_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>lockdowns</td>
      <td>1.016951</td>
      <td>940</td>
      <td>990</td>
    </tr>
    <tr>
      <th>1</th>
      <td>maskless</td>
      <td>0.996101</td>
      <td>118</td>
      <td>127</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sunsetting</td>
      <td>0.996084</td>
      <td>111</td>
      <td>119</td>
    </tr>
    <tr>
      <th>3</th>
      <td>childe</td>
      <td>0.980564</td>
      <td>209</td>
      <td>221</td>
    </tr>
    <tr>
      <th>4</th>
      <td>megalodon</td>
      <td>0.975273</td>
      <td>751</td>
      <td>792</td>
    </tr>
    <tr>
      <th>5</th>
      <td>newf</td>
      <td>0.962381</td>
      <td>107</td>
      <td>115</td>
    </tr>
    <tr>
      <th>6</th>
      <td>corona</td>
      <td>0.926739</td>
      <td>3553</td>
      <td>3684</td>
    </tr>
    <tr>
      <th>7</th>
      <td>filtrate</td>
      <td>0.918609</td>
      <td>102</td>
      <td>110</td>
    </tr>
    <tr>
      <th>8</th>
      <td>chaz</td>
      <td>0.899856</td>
      <td>190</td>
      <td>202</td>
    </tr>
    <tr>
      <th>9</th>
      <td>klee</td>
      <td>0.888728</td>
      <td>161</td>
      <td>173</td>
    </tr>
    <tr>
      <th>10</th>
      <td>rona</td>
      <td>0.886363</td>
      <td>409</td>
      <td>433</td>
    </tr>
    <tr>
      <th>11</th>
      <td>cerb</td>
      <td>0.869179</td>
      <td>315</td>
      <td>333</td>
    </tr>
    <tr>
      <th>12</th>
      <td>rittenhouse</td>
      <td>0.866206</td>
      <td>181</td>
      <td>194</td>
    </tr>
    <tr>
      <th>13</th>
      <td>vacuo</td>
      <td>0.862142</td>
      <td>170</td>
      <td>181</td>
    </tr>
    <tr>
      <th>14</th>
      <td>moderna</td>
      <td>0.843465</td>
      <td>199</td>
      <td>211</td>
    </tr>
    <tr>
      <th>15</th>
      <td>pandemic</td>
      <td>0.837175</td>
      <td>9504</td>
      <td>9957</td>
    </tr>
    <tr>
      <th>16</th>
      <td>spreader</td>
      <td>0.835257</td>
      <td>164</td>
      <td>175</td>
    </tr>
    <tr>
      <th>17</th>
      <td>distancing</td>
      <td>0.833737</td>
      <td>2910</td>
      <td>3038</td>
    </tr>
    <tr>
      <th>18</th>
      <td>sars</td>
      <td>0.827039</td>
      <td>880</td>
      <td>924</td>
    </tr>
    <tr>
      <th>19</th>
      <td>quarantines</td>
      <td>0.820280</td>
      <td>149</td>
      <td>160</td>
    </tr>
  </tbody>
</table>
</div>



Output semantic neologisms for inclusion in the paper.

```python
sem_change_cands_out = (sem_change_cands
    .nlargest(100, 'dist_sem')
    .assign(index_1 = lambda df: df.index + 1)
    .assign(dist_sem = lambda df: df['dist_sem'].round(2))
    .assign(dist_sem = lambda df: df['dist_sem'].apply('{:.2f}'.format))
    .rename({'index_1': '', 'lex': 'Lexeme', 'dist_sem': 'SemDist'}, axis=1)
)
```

```python
sem_change_cands_out.to_csv(
        f'{OUT_DIR}sem_change_cands.csv',
        columns=['', 'Lexeme', 'SemDist'],
        index=False
    )
```

### Inspect neighbourhood

Closer inspection of semantic neighbours between 2019 and 2020 for the term _distancing_. Unfortunately, due to the space limitation, these results had to be excluded from the paper.

```python
LEX_NBS = 'distancing'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=25,
    model_1=model_2019, 
    model_2=model_2020,
    k=10
)

display(
    nbs_model_1,
    nbs_model_2
)
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
      <th>Model</th>
      <th>Word</th>
      <th>SemDist</th>
      <th>Freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>distanced</td>
      <td>0.22</td>
      <td>309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>extricate</td>
      <td>0.27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>detaching</td>
      <td>0.34</td>
      <td>61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>disassociate</td>
      <td>0.34</td>
      <td>93</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>offing</td>
      <td>0.36</td>
      <td>48</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>recuse</td>
      <td>0.38</td>
      <td>50</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>recused</td>
      <td>0.40</td>
      <td>29</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>isolating</td>
      <td>0.42</td>
      <td>685</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>detach</td>
      <td>0.44</td>
      <td>245</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>distract</td>
      <td>0.45</td>
      <td>1553</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Model</th>
      <th>Word</th>
      <th>SemDist</th>
      <th>Freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>50601</th>
      <td>2</td>
      <td>distanced</td>
      <td>0.46</td>
      <td>326</td>
    </tr>
    <tr>
      <th>50602</th>
      <td>2</td>
      <td>isolation</td>
      <td>0.46</td>
      <td>2037</td>
    </tr>
    <tr>
      <th>50603</th>
      <td>2</td>
      <td>gatherings</td>
      <td>0.47</td>
      <td>921</td>
    </tr>
    <tr>
      <th>50604</th>
      <td>2</td>
      <td>distance</td>
      <td>0.48</td>
      <td>11355</td>
    </tr>
    <tr>
      <th>50605</th>
      <td>2</td>
      <td>lockdowns</td>
      <td>0.50</td>
      <td>990</td>
    </tr>
    <tr>
      <th>50606</th>
      <td>2</td>
      <td>quarantine</td>
      <td>0.53</td>
      <td>5225</td>
    </tr>
    <tr>
      <th>50607</th>
      <td>2</td>
      <td>masks</td>
      <td>0.53</td>
      <td>8997</td>
    </tr>
    <tr>
      <th>50608</th>
      <td>2</td>
      <td>quarantining</td>
      <td>0.53</td>
      <td>279</td>
    </tr>
    <tr>
      <th>50609</th>
      <td>2</td>
      <td>quarantines</td>
      <td>0.53</td>
      <td>160</td>
    </tr>
    <tr>
      <th>50610</th>
      <td>2</td>
      <td>lockdown</td>
      <td>0.53</td>
      <td>4642</td>
    </tr>
  </tbody>
</table>
</div>


```python
(nbs_model_1.filter(['Word', 'SemDist'])
	.to_csv(f'{OUT_DIR}nbs_{LEX_NBS}_2019.csv', float_format='%.2f', index=False))

(nbs_model_2.filter(['Word', 'SemDist'])
	.to_csv(f'{OUT_DIR}nbs_{LEX_NBS}_2020.csv', float_format='%.2f', index=False))
```

## Social semantic variation

### Covid-related communities

In this section, we determine those communities which are most actively engaged in Covid-related discourse.

#### read comments

```python
comments_dir_path = Path('../data/comments/lexeme/')
comments_paths = list(comments_dir_path.glob(f'Covid*.csv'))
```

```python
%%time
comments = read_comm_csvs(comments_paths)
comments
```

    CPU times: user 36.6 s, sys: 11.4 s, total: 48 s
    Wall time: 59 s





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
      <th>author</th>
      <th>body</th>
      <th>created_utc</th>
      <th>id</th>
      <th>subreddit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Gloob_Patrol</td>
      <td>I assume you work too so he's feeling like he ...</td>
      <td>2020-09-08 18:53:06</td>
      <td>g4guhl5</td>
      <td>LongDistance</td>
    </tr>
    <tr>
      <th>1</th>
      <td>amtrusc</td>
      <td>Strep swab and culture negative, I???m sure? Cou...</td>
      <td>2020-09-08 18:53:08</td>
      <td>g4guhsm</td>
      <td>tonsilstones</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ephuntz</td>
      <td>&amp;gt;Good point. My apologies. It's just becomi...</td>
      <td>2020-09-08 18:53:09</td>
      <td>g4guhua</td>
      <td>Winnipeg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>cstransfer</td>
      <td>Have you noticed an increase of people going e...</td>
      <td>2020-09-08 18:53:09</td>
      <td>g4guhu4</td>
      <td>financialindependence</td>
    </tr>
    <tr>
      <th>4</th>
      <td>IlliniWhoDat</td>
      <td>I haven't.  I have seen it online, but haven't...</td>
      <td>2020-09-08 18:53:13</td>
      <td>g4gui6o</td>
      <td>KoreanBeauty</td>
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
      <th>3800760</th>
      <td>willw</td>
      <td>Last group pre COVID!</td>
      <td>2020-07-01 21:59:48</td>
      <td>fwmqfbj</td>
      <td>jawsurgery</td>
    </tr>
    <tr>
      <th>3800761</th>
      <td>Daikataro</td>
      <td>If everyone is infected with COVID, new cases ...</td>
      <td>2020-07-01 21:59:49</td>
      <td>fwmqff2</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>3800762</th>
      <td>StabYourBloodIntoMe</td>
      <td>&amp;gt; If the mortality rate is actually decreas...</td>
      <td>2020-07-01 21:59:50</td>
      <td>fwmqfib</td>
      <td>dataisbeautiful</td>
    </tr>
    <tr>
      <th>3800763</th>
      <td>Shorse_rider</td>
      <td>I was a freelancer until covid and earned more...</td>
      <td>2020-07-01 21:59:55</td>
      <td>fwmqfuw</td>
      <td>AskWomen</td>
    </tr>
    <tr>
      <th>3800764</th>
      <td>Gayfetus</td>
      <td>This is actually fascinating and possibly incr...</td>
      <td>2020-07-01 21:59:57</td>
      <td>fwmqfz0</td>
      <td>Coronavirus</td>
    </tr>
  </tbody>
</table>
<p>3800765 rows ?? 5 columns</p>
</div>



#### get subreddit counts

```python
subr_counts = get_subr_counts(comments)
```

<a id='covid-communities'></a>

Top 15 communities that are most actively engaged in Covid-related discourse.

```python
subr_counts_plt = plot_subr_counts(subr_counts, k=15)
subr_counts_plt
```





<div id="altair-viz-4295c81803584537844c8caa048c68d3"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-4295c81803584537844c8caa048c68d3") {
      outputDiv = document.getElementById("altair-viz-4295c81803584537844c8caa048c68d3");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-ca557bf389e4a85e76234027511875f5"}, "mark": "bar", "encoding": {"x": {"field": "comments_num", "type": "quantitative"}, "y": {"field": "subreddit", "sort": "-x", "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-ca557bf389e4a85e76234027511875f5": [{"subreddit": "Coronavirus", "comments_num": 173397}, {"subreddit": "AskReddit", "comments_num": 142358}, {"subreddit": "politics", "comments_num": 136270}, {"subreddit": "worldnews", "comments_num": 87336}, {"subreddit": "news", "comments_num": 55561}, {"subreddit": "conspiracy", "comments_num": 52780}, {"subreddit": "wallstreetbets", "comments_num": 40424}, {"subreddit": "DDnews", "comments_num": 37062}, {"subreddit": "AmItheAsshole", "comments_num": 35861}, {"subreddit": "pics", "comments_num": 27597}, {"subreddit": "ukpolitics", "comments_num": 25970}, {"subreddit": "LockdownSkepticism", "comments_num": 22906}, {"subreddit": "PublicFreakout", "comments_num": 22891}, {"subreddit": "canada", "comments_num": 22754}, {"subreddit": "Conservative", "comments_num": 21673}]}}, {"mode": "vega-lite"});
</script>



```python
subr_counts_plt.save(f'{OUT_DIR}subr_counts.png', scale_factor=2.0)
```

### Train models

In this section, we train community-specific embedding models.

```python
SUBR = 'Coronavirus'
```

```python
fpaths = get_comments_paths_subr(COMMENTS_DIR_SUBR, SUBR)
comments = read_comm_csvs(fpaths)
```

```python
%%time
comments_clean = clean_comments(comments)
```

    conv_to_lowerc       (4121144, 5) 0:00:08.279838      
    rm_punct             (4121144, 5) 0:00:31.917256      
    tokenize             (4121144, 5) 0:07:40.929735      
    rem_short_comments   (2927221, 5) 0:01:04.440039      
    CPU times: user 1min 21s, sys: 3min 17s, total: 4min 38s
    Wall time: 10min 42s


```python
docs = comments_clean['body']
docs = docs.to_list()
```

```python
with open(f'{OUT_DIR}docs_clean/{SUBR}.pickle', 'wb') as fp:
    pickle.dump(docs, fp)
```

Load pre-processed comments from disk.

```python
with open(f'{OUT_DIR}docs_clean/{SUBR}.pickle', 'rb') as fp:
    docs = pickle.load(fp)
```

```python
f'{len(docs):,}'
```




    '2,927,221'



```python
corpus = Corpus(docs)
```

```python
%%time
model = train_model(corpus, EPOCHS=20)
```

    CPU times: user 21min 15s, sys: 10.7 s, total: 21min 26s
    Wall time: 4min 44s


Print vocabulary size.

```python
f'{len(model.wv.key_to_index):,}'
```




    '38,558'



```python
model.save(f'{OUT_DIR}models/{SUBR}.model')
```

### Load models

```python
model_names = ['Coronavirus', 'conspiracy']
```

```python
models = []
for name in model_names:
	model = make_model_dict(name)
	model['model'] = Word2Vec.load(model['path'])
	models.append(model)
```

### Align models

```python
for model in models:
	model['vocab'] = len(model['model'].wv.key_to_index)
```

```python
smart_procrustes_align_gensim(models[0]['model'], models[1]['model'])
```

    67181 67181
    67181 67181





    <gensim.models.word2vec.Word2Vec at 0x172c82530>



```python
assert len(models[0]['model'].wv.key_to_index) == len(models[1]['model'].wv.key_to_index)
```

```python
models_vocab = (pd.DataFrame(models)
	.filter(['name', 'vocab'])
	.rename({'name': 'Model', 'vocab': 'Words'}, axis=1)
)

models_vocab
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
      <th>Model</th>
      <th>Words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Coronavirus</td>
      <td>94816</td>
    </tr>
    <tr>
      <th>1</th>
      <td>conspiracy</td>
      <td>112599</td>
    </tr>
  </tbody>
</table>
</div>



```python
# models_vocab.to_csv(f'../out/vocabs/vocab_{models[0]["name"]}--{models[1]["name"]}.csv', index=False)
```

### Semantic neighbourhoods

```python
distances = measure_distances(models[0]['model'], models[1]['model'])
```

#### words that differ the most between both communities

Due to space limitations, the following results had to be excluded from the paper.

```python
blacklist_lex = load_blacklist_lex()

k = 20
freq_min = 100

sem_change_cands = (distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')
    .query('lex.str.isalpha() == True')
    .query('lex.str.len() > 3')
    .query('lex not in @blacklist_lex')
    .nlargest(k, 'dist_sem')
    .reset_index(drop=True)
)

sem_change_cands
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
      <th>lex</th>
      <th>dist_sem</th>
      <th>freq_1</th>
      <th>freq_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>soliciting</td>
      <td>1.003643</td>
      <td>2233</td>
      <td>2474</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nimrod</td>
      <td>0.974182</td>
      <td>103</td>
      <td>144</td>
    </tr>
    <tr>
      <th>2</th>
      <td>incivility</td>
      <td>0.958038</td>
      <td>16347</td>
      <td>15690</td>
    </tr>
    <tr>
      <th>3</th>
      <td>globes</td>
      <td>0.955581</td>
      <td>140</td>
      <td>193</td>
    </tr>
    <tr>
      <th>4</th>
      <td>submitter</td>
      <td>0.952117</td>
      <td>261</td>
      <td>352</td>
    </tr>
    <tr>
      <th>5</th>
      <td>acorn</td>
      <td>0.950665</td>
      <td>105</td>
      <td>148</td>
    </tr>
    <tr>
      <th>6</th>
      <td>subsequently</td>
      <td>0.946088</td>
      <td>12174</td>
      <td>11956</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resubmit</td>
      <td>0.937007</td>
      <td>1763</td>
      <td>1927</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mouthy</td>
      <td>0.934621</td>
      <td>129</td>
      <td>178</td>
    </tr>
    <tr>
      <th>9</th>
      <td>narc</td>
      <td>0.930710</td>
      <td>224</td>
      <td>305</td>
    </tr>
    <tr>
      <th>10</th>
      <td>spook</td>
      <td>0.920622</td>
      <td>317</td>
      <td>423</td>
    </tr>
    <tr>
      <th>11</th>
      <td>nicaraguan</td>
      <td>0.914530</td>
      <td>111</td>
      <td>154</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yadda</td>
      <td>0.913075</td>
      <td>184</td>
      <td>252</td>
    </tr>
    <tr>
      <th>13</th>
      <td>amputatorbot</td>
      <td>0.905838</td>
      <td>9444</td>
      <td>9450</td>
    </tr>
    <tr>
      <th>14</th>
      <td>anons</td>
      <td>0.901062</td>
      <td>259</td>
      <td>351</td>
    </tr>
    <tr>
      <th>15</th>
      <td>mcnamara</td>
      <td>0.899120</td>
      <td>108</td>
      <td>150</td>
    </tr>
    <tr>
      <th>16</th>
      <td>editorialization</td>
      <td>0.895667</td>
      <td>1182</td>
      <td>1362</td>
    </tr>
    <tr>
      <th>17</th>
      <td>spooks</td>
      <td>0.890019</td>
      <td>263</td>
      <td>354</td>
    </tr>
    <tr>
      <th>18</th>
      <td>durham</td>
      <td>0.889415</td>
      <td>490</td>
      <td>610</td>
    </tr>
    <tr>
      <th>19</th>
      <td>nist</td>
      <td>0.889119</td>
      <td>204</td>
      <td>277</td>
    </tr>
  </tbody>
</table>
</div>



```python
sem_change_cands_out = (sem_change_cands
    .nlargest(100, 'dist_sem')
    .assign(index_1 = lambda df: df.index + 1)
    .assign(dist_sem = lambda df: df['dist_sem'].round(2))
    .assign(dist_sem = lambda df: df['dist_sem'].apply('{:.2f}'.format))
    .rename({'index_1': '', 'lex': 'Lexeme', 'dist_sem': 'SemDist'}, axis=1)
    .filter(['Lexeme', 'SemDist'])
)
sem_change_cands_out

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
      <th>Lexeme</th>
      <th>SemDist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>soliciting</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nimrod</td>
      <td>0.97</td>
    </tr>
    <tr>
      <th>2</th>
      <td>incivility</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>globes</td>
      <td>0.96</td>
    </tr>
    <tr>
      <th>4</th>
      <td>submitter</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>acorn</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>6</th>
      <td>subsequently</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>7</th>
      <td>resubmit</td>
      <td>0.94</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mouthy</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>9</th>
      <td>narc</td>
      <td>0.93</td>
    </tr>
    <tr>
      <th>10</th>
      <td>spook</td>
      <td>0.92</td>
    </tr>
    <tr>
      <th>11</th>
      <td>nicaraguan</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>12</th>
      <td>yadda</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>13</th>
      <td>amputatorbot</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>14</th>
      <td>anons</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>15</th>
      <td>mcnamara</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>16</th>
      <td>editorialization</td>
      <td>0.90</td>
    </tr>
    <tr>
      <th>17</th>
      <td>spooks</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>18</th>
      <td>durham</td>
      <td>0.89</td>
    </tr>
    <tr>
      <th>19</th>
      <td>nist</td>
      <td>0.89</td>
    </tr>
  </tbody>
</table>
</div>



```python
sem_change_cands_out.to_csv(
        f'{OUT_DIR}sem_var_soc_cands.csv',
        index=False
    )
```

#### nearest neighbours for target lexemes in both communities

```python
LEX_NBS = 'vaccines'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=100,
    model_1=models[0]['model'], 
    model_2=models[1]['model'],
    k=10
)

display(
    nbs_model_1,
    nbs_model_2
)
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
      <th>Model</th>
      <th>Word</th>
      <th>SemDist</th>
      <th>Freq</th>
      <th>vec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>vaccine</td>
      <td>0.25</td>
      <td>109094</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>vaccinations</td>
      <td>0.31</td>
      <td>3305</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>antivirals</td>
      <td>0.38</td>
      <td>357</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>treatments</td>
      <td>0.40</td>
      <td>4737</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>drugs</td>
      <td>0.41</td>
      <td>13655</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>therapies</td>
      <td>0.42</td>
      <td>425</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>doses</td>
      <td>0.43</td>
      <td>6558</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>trials</td>
      <td>0.45</td>
      <td>10095</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>strains</td>
      <td>0.46</td>
      <td>3875</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>therapeutics</td>
      <td>0.46</td>
      <td>385</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Model</th>
      <th>Word</th>
      <th>SemDist</th>
      <th>Freq</th>
      <th>vec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21542</th>
      <td>2</td>
      <td>vaccinations</td>
      <td>0.20</td>
      <td>3624</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21543</th>
      <td>2</td>
      <td>vaccine</td>
      <td>0.23</td>
      <td>112485</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21544</th>
      <td>2</td>
      <td>vaccination</td>
      <td>0.36</td>
      <td>7780</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21545</th>
      <td>2</td>
      <td>treatments</td>
      <td>0.38</td>
      <td>4874</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21546</th>
      <td>2</td>
      <td>medications</td>
      <td>0.40</td>
      <td>1614</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21547</th>
      <td>2</td>
      <td>vax</td>
      <td>0.42</td>
      <td>3208</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21548</th>
      <td>2</td>
      <td>injections</td>
      <td>0.43</td>
      <td>795</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21549</th>
      <td>2</td>
      <td>adjuvants</td>
      <td>0.43</td>
      <td>199</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21550</th>
      <td>2</td>
      <td>medicines</td>
      <td>0.43</td>
      <td>1208</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
    <tr>
      <th>21551</th>
      <td>2</td>
      <td>viruses</td>
      <td>0.45</td>
      <td>17105</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
    </tr>
  </tbody>
</table>
</div>


#### biggest discrepancies in nearest neighbours for target lexemes

```python
lex = 'vaccines'
topn = 15

nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=lex, 
    freq_min=100,
    model_1=models[0]['model'], 
    model_2=models[1]['model'],
    k=100_000
)

nbs_diffs = pd.merge(
    nbs_model_1, nbs_model_2, 
    on='Word',
    suffixes = ('_1', '_2')
)

nbs_diffs = nbs_diffs\
    .assign(sim_diff = abs(nbs_diffs['SemDist_1'] - nbs_diffs['SemDist_2']))\
    .sort_values('sim_diff', ascending=True)\
    .reset_index(drop=True)\
    .query('Word.str.len() >= 4')

subr_1_nbs = nbs_diffs\
    .query('SemDist_1 < SemDist_2')\
    .nlargest(topn, 'sim_diff')

subr_2_nbs = nbs_diffs\
    .query('SemDist_2 < SemDist_1')\
    .nlargest(topn, 'sim_diff')

display(subr_1_nbs, subr_2_nbs)
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
      <th>Model_1</th>
      <th>Word</th>
      <th>SemDist_1</th>
      <th>Freq_1</th>
      <th>vec_1</th>
      <th>Model_2</th>
      <th>SemDist_2</th>
      <th>Freq_2</th>
      <th>vec_2</th>
      <th>sim_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21540</th>
      <td>1</td>
      <td>candidates</td>
      <td>0.48</td>
      <td>4842</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.82</td>
      <td>4925</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.34</td>
    </tr>
    <tr>
      <th>21539</th>
      <td>1</td>
      <td>dyson</td>
      <td>0.77</td>
      <td>114</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.10</td>
      <td>158</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>21536</th>
      <td>1</td>
      <td>parallel</td>
      <td>0.80</td>
      <td>943</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.09</td>
      <td>1095</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>21530</th>
      <td>1</td>
      <td>lamp</td>
      <td>0.86</td>
      <td>224</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.14</td>
      <td>305</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>21531</th>
      <td>1</td>
      <td>underworld</td>
      <td>0.86</td>
      <td>115</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.14</td>
      <td>159</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>21526</th>
      <td>1</td>
      <td>oxford</td>
      <td>0.64</td>
      <td>4128</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.91</td>
      <td>4378</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>21519</th>
      <td>1</td>
      <td>slices</td>
      <td>0.81</td>
      <td>113</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.07</td>
      <td>157</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>21509</th>
      <td>1</td>
      <td>fade</td>
      <td>0.87</td>
      <td>490</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.12</td>
      <td>611</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21504</th>
      <td>1</td>
      <td>sputnik</td>
      <td>0.68</td>
      <td>279</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.93</td>
      <td>376</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21507</th>
      <td>1</td>
      <td>approved</td>
      <td>0.64</td>
      <td>7276</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.89</td>
      <td>7443</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21508</th>
      <td>1</td>
      <td>preprints</td>
      <td>0.63</td>
      <td>177</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.88</td>
      <td>242</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>21484</th>
      <td>1</td>
      <td>candidate</td>
      <td>0.71</td>
      <td>8515</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.95</td>
      <td>8391</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>21485</th>
      <td>1</td>
      <td>operative</td>
      <td>0.97</td>
      <td>542</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.21</td>
      <td>670</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>21486</th>
      <td>1</td>
      <td>designation</td>
      <td>0.80</td>
      <td>191</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>1.04</td>
      <td>261</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.24</td>
    </tr>
    <tr>
      <th>21489</th>
      <td>1</td>
      <td>russians</td>
      <td>0.70</td>
      <td>2919</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.94</td>
      <td>3184</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.24</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>Model_1</th>
      <th>Word</th>
      <th>SemDist_1</th>
      <th>Freq_1</th>
      <th>vec_1</th>
      <th>Model_2</th>
      <th>SemDist_2</th>
      <th>Freq_2</th>
      <th>vec_2</th>
      <th>sim_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21541</th>
      <td>1</td>
      <td>gmos</td>
      <td>0.85</td>
      <td>130</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.49</td>
      <td>179</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>21537</th>
      <td>1</td>
      <td>mandated</td>
      <td>1.09</td>
      <td>2456</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.79</td>
      <td>2732</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>21538</th>
      <td>1</td>
      <td>disrespecting</td>
      <td>1.25</td>
      <td>231</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.95</td>
      <td>314</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.30</td>
    </tr>
    <tr>
      <th>21534</th>
      <td>1</td>
      <td>neuralink</td>
      <td>0.98</td>
      <td>210</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.69</td>
      <td>285</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>21535</th>
      <td>1</td>
      <td>vaxx</td>
      <td>0.86</td>
      <td>633</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.57</td>
      <td>771</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.29</td>
    </tr>
    <tr>
      <th>21532</th>
      <td>1</td>
      <td>poisons</td>
      <td>0.84</td>
      <td>171</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.56</td>
      <td>234</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.28</td>
    </tr>
    <tr>
      <th>21524</th>
      <td>1</td>
      <td>preventable</td>
      <td>1.02</td>
      <td>1550</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.75</td>
      <td>1726</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>21525</th>
      <td>1</td>
      <td>mandating</td>
      <td>1.08</td>
      <td>840</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.81</td>
      <td>998</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>21527</th>
      <td>1</td>
      <td>sugar</td>
      <td>0.99</td>
      <td>2478</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.72</td>
      <td>2747</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>21528</th>
      <td>1</td>
      <td>leukemia</td>
      <td>0.90</td>
      <td>140</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.63</td>
      <td>193</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>21522</th>
      <td>1</td>
      <td>eugenicist</td>
      <td>1.11</td>
      <td>195</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.85</td>
      <td>265</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>21523</th>
      <td>1</td>
      <td>cidrap</td>
      <td>1.08</td>
      <td>330</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.82</td>
      <td>437</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>21511</th>
      <td>1</td>
      <td>mandatory</td>
      <td>1.01</td>
      <td>9592</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.75</td>
      <td>9516</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>21512</th>
      <td>1</td>
      <td>lice</td>
      <td>1.00</td>
      <td>136</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.74</td>
      <td>187</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.26</td>
    </tr>
    <tr>
      <th>21513</th>
      <td>1</td>
      <td>cv19</td>
      <td>0.99</td>
      <td>539</td>
      <td>[1.6486416, -0.77526903, -0.25844133, -2.18959...</td>
      <td>2</td>
      <td>0.73</td>
      <td>667</td>
      <td>[0.34145182, -1.5721449, -0.045296144, -1.6733...</td>
      <td>0.26</td>
    </tr>
  </tbody>
</table>
</div>


### Maps of social semantic variation

<a id='sem-maps'></a>

The following section contains the plots for Figure 3.

```python
lex = 'vaccines'
```

```python
nbs_vecs = pd.concat([get_nbs_vecs(lex, model, k=750) for model in models])
```

#### common neighbours

```python
#data
nbs_vecs = dim_red_nbs_vecs(nbs_vecs, perplexity=0)
```

    /Users/quirin/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:982: FutureWarning: The PCA initialization in TSNE will change to have the standard deviation of PC1 equal to 1e-4 in 1.2. This will ensure better convergence.
      warnings.warn(


```python
#data
nbs_sim = (nbs_vecs
	.groupby('subreddit')
	.apply(lambda df: df.nlargest(10, 'sim'))
	.reset_index(drop=True)
)
```

```python
#data
chart_sims = (alt.Chart(nbs_sim).mark_text().encode(
		x='x_tsne:Q',
		y='y_tsne:Q',
		text='lex',
		color='subreddit:N'
	))

chart_sims
```



<div id="altair-viz-917d17b736c844e9bf42e2f8bb87f17e"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-917d17b736c844e9bf42e2f8bb87f17e") {
      outputDiv = document.getElementById("altair-viz-917d17b736c844e9bf42e2f8bb87f17e");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-9ea6865f0b0a7110d19a42d8a3058e49"}, "mark": "text", "encoding": {"color": {"field": "subreddit", "type": "nominal"}, "text": {"field": "lex", "type": "nominal"}, "x": {"field": "x_tsne", "type": "quantitative"}, "y": {"field": "y_tsne", "type": "quantitative"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-9ea6865f0b0a7110d19a42d8a3058e49": [{"lex": "vaccine", "type": "nb", "subreddit": "Coronavirus", "vec": [1.240659475326538, -0.12387722730636597, 2.0859854221343994, -2.255932569503784, -2.4089362621307373, 1.113614559173584, 1.5809203386306763, -0.3034607172012329, -0.07281932234764099, 0.4294254779815674, 2.127032995223999, 2.700216770172119, 6.618000507354736, -2.335289716720581, 3.1877694129943848, 1.2342853546142578, 2.7840476036071777, -1.7046575546264648, 1.2613478899002075, 1.9026646614074707, -3.67684006690979, -1.2600479125976562, -1.6607239246368408, 1.6806471347808838, 1.6238232851028442, -0.33804723620414734, 1.370609998703003, 1.5215386152267456, 3.849821090698242, -1.5824761390686035, 0.8997215032577515, -2.205965995788574, 0.06630970537662506, -1.5000531673431396, 2.654740810394287, 0.7006876468658447, -2.129399538040161, 1.156707763671875, -3.7727625370025635, 0.9638364315032959, 1.062569499015808, -0.34880363941192627, 0.35328158736228943, 0.5716593265533447, 4.205084323883057, -0.9088820815086365, 1.657668113708496, 0.48014888167381287, -0.7871667146682739, -0.23496223986148834, -0.019358498975634575, -0.4234621524810791, 1.5934861898422241, 1.1553852558135986, -2.540015459060669, -0.3005439043045044, 1.9365246295928955, -2.7002835273742676, -1.805703043937683, -0.19651713967323303, 1.5991886854171753, 3.5201218128204346, -0.6797454357147217, 1.6274778842926025, 2.0237982273101807, -2.970956802368164, -5.0459113121032715, 1.5014322996139526, -1.8127299547195435, 1.4604676961898804, 0.4915148913860321, -3.5254690647125244, 0.02814287319779396, 0.49888691306114197, 0.2185327708721161, -6.465264797210693, 2.078871726989746, 1.260628342628479, -2.480525493621826, 1.0336451530456543, -1.1876598596572876, -2.092618942260742, -0.42990097403526306, 0.9167951941490173, -4.241758346557617, 0.08029661327600479, -1.6456680297851562, -2.183732748031616, -1.6483442783355713, 4.724308013916016, 0.6142719388008118, 2.4626123905181885, 1.9064592123031616, 2.9948220252990723, -1.517002820968628, -3.4840097427368164, -0.9298512935638428, 4.900462627410889, 0.4240912199020386, -0.43076291680336, 1.5202562808990479, -1.0017470121383667, -0.24758845567703247, -1.015495777130127, -2.7533464431762695, -4.279035568237305, 1.8416907787322998, -4.671482086181641, 0.5221771001815796, 3.0537195205688477, -2.457305908203125, -5.231683254241943, -1.5582433938980103, -1.256283164024353, 0.49324145913124084, 0.9513742923736572, 2.439502716064453, -2.3829703330993652, 0.06406519562005997, -2.084829330444336, 0.6579539179801941, -4.0658860206604, 2.495007276535034, 2.1900675296783447, 0.8470639586448669, 4.430922508239746, 0.19597843289375305, -4.593856334686279, -0.6973477602005005, -0.8052508234977722, 2.5819456577301025, 0.06459618359804153, -1.1414448022842407, -4.6662917137146, 3.2665138244628906, -0.5961626768112183, -2.501999616622925, -1.0064895153045654, -1.902847409248352, -1.2036824226379395, 0.8519977331161499, -2.9120941162109375, 0.18145795166492462, 2.713224411010742, 1.1505610942840576, -3.2271814346313477, -0.3960990905761719, -3.286733388900757, 4.040095329284668, 3.482245683670044, 0.5294197797775269, -2.109053134918213, -1.1691378355026245, -2.6012911796569824, -0.846846342086792, -3.569767713546753, -3.354424238204956, 2.564056158065796, -1.447676420211792, 1.9255346059799194, 1.2710307836532593, 2.940152406692505, 4.116117477416992, 2.3450236320495605, -0.4463758170604706, 0.11277039349079132, 0.8292954564094543, 1.8342010974884033, -0.5350095629692078, 2.2307605743408203, -3.254854202270508, 0.2633203864097595, -2.6251111030578613, 0.3934183418750763, -0.554554283618927, -1.6297205686569214, -1.8385133743286133, 1.0843091011047363, -3.3356752395629883, 2.3924741744995117, -2.2508068084716797, 1.6433364152908325, -0.06707443296909332, -2.12490177154541, -0.40552666783332825, -3.525416851043701, 4.489255428314209, -0.5862698554992676, 0.5568583607673645, -4.6364426612854, 0.02129187248647213, 0.24796977639198303, -0.6130716800689697, 1.2851543426513672, -2.5074515342712402, -0.6874294877052307, -0.46586301922798157, -4.11159610748291, -0.6338808536529541, -0.1439998745918274, -3.12394118309021, 2.1024410724639893, -0.07824348658323288, 4.101337432861328, -3.255927562713623, 1.6419880390167236, -1.1096116304397583, 2.013307571411133, 2.890766143798828, -0.18399211764335632, 2.3225719928741455, -1.423804759979248, -2.1818768978118896, 2.360349416732788, 2.6803324222564697, -1.480281949043274, -3.524153709411621, -2.5288076400756836, 0.8279819488525391, -2.953686237335205, -0.025147628039121628, 2.6025400161743164, 0.16853608191013336, -1.3117636442184448, -4.058568477630615, -0.7796360850334167, 0.2039490044116974, 1.3200980424880981, 1.0242420434951782, -0.6836625933647156, 1.4339666366577148, -1.8513661623001099, -0.44087815284729004, -1.199843168258667, 3.2136149406433105, 0.9411898851394653, -2.595226526260376, 3.268876552581787, -1.946613073348999, -0.6734466552734375, -0.4573156535625458, -2.204773426055908, 1.1167967319488525, -2.25915265083313, 0.029236512258648872, 3.1625492572784424, 2.122859001159668, 3.0529468059539795, -2.5641207695007324, -0.5742003321647644, 1.981998085975647, -0.3789007067680359, 3.4942269325256348, 1.362125039100647, -0.7571232914924622, 1.9467066526412964, -0.7270846366882324, 0.17168830335140228, -2.332401752471924, 1.428552508354187, -1.8193589448928833, -0.275041788816452, 0.1623024344444275, -3.4050915241241455, 0.8814846277236938, -1.0797128677368164, 0.9680081009864807, -2.2828116416931152, -2.9074885845184326, -1.8425389528274536, 1.8144457340240479, -1.665337085723877, 2.757113218307495, 1.2409206628799438, -2.6465625762939453, 1.6102749109268188, -0.8410540819168091, 5.9611945152282715, -1.0826398134231567, -2.5621612071990967, -1.8032498359680176, -1.498968243598938, 3.510639190673828, -0.6040961146354675, 2.128664970397949, -0.39729705452919006, 3.0838165283203125, -0.1046680435538292, -1.5005815029144287, -0.21370963752269745, 1.0441924333572388, 1.2818082571029663, -0.7427034974098206, 2.7568256855010986, 0.07565765827894211, -0.1281929463148117, -2.6360161304473877, 1.1883718967437744, -2.9151341915130615, 1.2181164026260376], "sim": 0.7498504519462585, "freq": 109094.0, "x_tsne": 78.3118667602539, "y_tsne": -139.80377197265625}, {"lex": "vaccinations", "type": "nb", "subreddit": "Coronavirus", "vec": [0.14529788494110107, -0.560935378074646, 1.5990644693374634, 0.33777764439582825, -0.06309930980205536, 1.007271647453308, 0.7137916684150696, -2.238698720932007, 0.6142995357513428, 0.31784898042678833, 1.895662784576416, 1.3464406728744507, 2.50917911529541, -0.787909209728241, 2.661410093307495, 2.317296266555786, 0.08340365439653397, -1.7278169393539429, -0.013478312641382217, -0.09841904044151306, -1.6405400037765503, 0.328628271818161, -2.1302852630615234, 0.9052079319953918, 0.8118285536766052, -0.36906859278678894, 0.9131327271461487, -0.3056502044200897, -0.9032238721847534, 0.40169933438301086, 0.6211521029472351, -0.22208289802074432, 2.6262142658233643, -2.4167776107788086, 0.8041032552719116, -0.30208033323287964, -1.3956432342529297, -1.5479621887207031, -2.8472046852111816, -1.1658287048339844, -2.31609845161438, -0.2014741599559784, -0.9729896187782288, 0.22782699763774872, 0.42592132091522217, -0.27810409665107727, 0.6691171526908875, 0.3276492655277252, 1.5110543966293335, -2.120708465576172, -0.8225938081741333, 0.11045854538679123, -1.697651982307434, 1.1478023529052734, -2.4285941123962402, -1.1063013076782227, -0.3612861931324005, -0.7924422025680542, 0.9855408072471619, -0.7581987380981445, 0.0005478572566062212, 1.8575624227523804, -0.24837714433670044, 1.5451254844665527, 1.2856181859970093, 0.11010023206472397, -1.3104498386383057, 0.03672530874609947, -0.6930043697357178, 1.3828967809677124, -1.4848533868789673, -2.6897470951080322, -0.9630459547042847, 1.3351752758026123, -0.49568435549736023, -3.4180893898010254, 0.6704135537147522, -0.4803704619407654, 1.1557615995407104, -0.5393789410591125, 1.4218335151672363, -1.5334620475769043, -1.026323914527893, 0.4456726312637329, -1.8037915229797363, 1.202203631401062, -1.3037774562835693, 0.032238543033599854, 1.1556471586227417, 0.7708377242088318, 0.17202815413475037, 0.29104143381118774, 2.866347551345825, 3.6189444065093994, -1.4232895374298096, -1.9901344776153564, -0.7175313234329224, 0.7333604693412781, 0.43106552958488464, -1.4486241340637207, 2.5788090229034424, -0.5688166618347168, -1.7254798412322998, -2.663191318511963, 0.4864741861820221, -2.9342899322509766, 1.0731834173202515, -1.0428882837295532, 0.8991574645042419, 0.7337362170219421, -0.9153167009353638, -3.3734922409057617, -1.0476620197296143, -1.4365599155426025, -0.7484012842178345, -0.9162548780441284, -2.1181509494781494, 2.2275266647338867, -0.4479675590991974, 0.15293166041374207, 0.6091989874839783, -1.9822295904159546, 1.6709874868392944, 1.0519033670425415, 0.4564582407474518, 3.458683490753174, -0.2904324233531952, 0.9340646862983704, -1.0016642808914185, -1.0340546369552612, 0.2873338460922241, 0.43579939007759094, -1.0625593662261963, -1.1918972730636597, 2.8826866149902344, -0.4102255702018738, 0.24945899844169617, 0.27292904257774353, -1.3760188817977905, 0.5184083580970764, 0.8827694654464722, -2.074819564819336, 0.6726381182670593, -0.1115444004535675, 0.7872427105903625, -2.0899906158447266, 0.9526763558387756, -1.6812423467636108, 2.0100200176239014, -1.3512064218521118, -0.5420308709144592, -0.8136899471282959, -0.9999071359634399, -1.7543253898620605, -1.4037505388259888, -1.485615611076355, -1.3815003633499146, 0.3182651996612549, -1.3890935182571411, -0.6496326923370361, -0.9073998928070068, 1.6360015869140625, 1.2721494436264038, 0.7023478746414185, 1.3358474969863892, 1.0542445182800293, 0.4740581214427948, 3.283874988555908, -0.6851761937141418, 0.8599848747253418, -0.6889975070953369, 0.5248411297798157, -0.7982285022735596, 0.7839146852493286, -0.36241164803504944, 0.09969513863325119, -2.556257486343384, -0.8822682499885559, -0.5369375944137573, -0.6382874250411987, -0.47490644454956055, 1.1540788412094116, -0.2479703277349472, -1.214956521987915, 0.08786817640066147, 0.05955476686358452, 2.2603888511657715, -2.2066612243652344, 0.7361710071563721, -0.9241895079612732, -2.324049472808838, 0.779872477054596, 1.422406554222107, 0.8302819132804871, -1.1372185945510864, -1.2906397581100464, -2.9865784645080566, -1.3263499736785889, -0.36424386501312256, 0.8014441728591919, -0.79006427526474, -0.11772165447473526, -0.78876131772995, 4.0580949783325195, -3.433959484100342, -0.7089949250221252, 0.2796586751937866, 0.386523574590683, -0.3621343970298767, -0.7138134241104126, 1.969287395477295, 0.3486623167991638, -0.9082993268966675, 1.6996299028396606, 0.25074222683906555, 2.400075674057007, 0.4138803482055664, -2.4427802562713623, 1.194222092628479, -0.6000868678092957, -1.0149903297424316, 0.8685919046401978, 0.9533609747886658, -0.6461541652679443, -1.464830756187439, -1.1373909711837769, 0.7328101992607117, 0.7264276742935181, 1.7782706022262573, -0.9486269354820251, 0.9085759520530701, -1.0678263902664185, -0.6241902112960815, 0.2722194492816925, 0.9292005896568298, 0.6178733110427856, -0.8142511248588562, 3.1437673568725586, -2.265713691711426, 0.6351848244667053, 0.0500536784529686, -0.01866026222705841, 0.40079566836357117, -1.1777842044830322, -0.2879790961742401, 2.5226993560791016, 2.23030424118042, -0.8713626861572266, 1.1709634065628052, -0.7795864939689636, 0.7165480852127075, -1.0415961742401123, -0.5767276287078857, 1.0207180976867676, 1.5672234296798706, 0.6773965358734131, 0.9392650723457336, -0.19946174323558807, -2.163386583328247, -0.36105677485466003, 0.0555863231420517, 1.6834824085235596, -0.1715308427810669, -1.4879838228225708, 1.4693870544433594, 1.2088522911071777, -0.9271941781044006, -1.690025806427002, -2.0935840606689453, -0.09803417325019836, 0.13703201711177826, -2.975574254989624, 0.002818366279825568, -0.26817017793655396, 0.6652260422706604, -1.297634482383728, -1.269129991531372, 2.8126840591430664, -0.6405746936798096, -1.409275770187378, -1.557133436203003, -0.1636580377817154, 0.8426895141601562, 2.15389084815979, 1.8217036724090576, -2.530918598175049, -0.1130203977227211, -1.0796946287155151, -1.7614558935165405, 0.9448041319847107, -0.37405896186828613, 0.9832202792167664, 1.7214943170547485, 0.2009585201740265, 0.3137742877006531, 0.20409056544303894, -0.3910960555076599, -2.63962721824646, -0.32316097617149353, -0.9837557077407837], "sim": 0.6917760372161865, "freq": 3305.0, "x_tsne": 83.31448364257812, "y_tsne": 2.2829763889312744}, {"lex": "antivirals", "type": "nb", "subreddit": "Coronavirus", "vec": [1.2559727430343628, -1.6036319732666016, -0.10239306092262268, -1.4454189538955688, -1.7006831169128418, 2.260450839996338, 1.742474913597107, -0.19135090708732605, -0.6935892105102539, -1.9388395547866821, 0.7069315314292908, 0.13862097263336182, 2.6774723529815674, -0.9233065843582153, 2.4507482051849365, 0.8615460991859436, -0.0762847289443016, -0.8841625452041626, 3.157545328140259, 0.5305265188217163, -1.423203468322754, 2.3234846591949463, -0.5390329957008362, -0.28873008489608765, 3.7255027294158936, 0.5548035502433777, 0.26409733295440674, 0.08346867561340332, 0.15552400052547455, -1.5083266496658325, 0.11496049910783768, -0.15785139799118042, -0.25693443417549133, -1.6929759979248047, 1.5621294975280762, -0.13416893780231476, -2.0681896209716797, -1.0305756330490112, 1.4982818365097046, -0.42115381360054016, -0.908869743347168, 1.8730638027191162, -0.7550492882728577, 1.4660611152648926, 0.7565915584564209, 0.3955504894256592, 1.5276991128921509, 1.4751927852630615, 0.650058925151825, 0.41424867510795593, -0.36090442538261414, -0.5198388695716858, -0.7314525842666626, 0.5508855581283569, -3.112006187438965, 0.811448872089386, 1.2039259672164917, 0.1592472493648529, -1.2005290985107422, -3.1252665519714355, 0.7636426687240601, 2.2101824283599854, -1.2633575201034546, 1.9130573272705078, 0.4814314544200897, -0.48464107513427734, 0.6407943367958069, 1.5420788526535034, 0.1395912915468216, 0.7784578204154968, 1.2500824928283691, -0.2944427728652954, -0.2032831758260727, 1.1912024021148682, -1.4725548028945923, -1.3238641023635864, -0.5058472156524658, 0.042711563408374786, -1.453364372253418, 1.3102984428405762, -0.20786869525909424, 0.32357126474380493, -0.14752265810966492, -0.8155977725982666, -1.393315076828003, 0.4006984233856201, -0.7855187058448792, 0.19087886810302734, -0.3208209276199341, 1.552613615989685, -1.9957619905471802, -1.9158164262771606, 0.7412514686584473, 1.1558635234832764, -2.728759765625, -0.7749704122543335, 0.5688632130622864, -0.8832592368125916, 0.6458104252815247, 0.12446244806051254, -0.36402660608291626, 0.46872448921203613, -0.31835922598838806, -0.3548961281776428, -1.0353736877441406, -2.02695369720459, 0.8401811718940735, -0.5742985606193542, -0.1275564432144165, -1.3591859340667725, 1.7726627588272095, -1.542466640472412, -1.3255946636199951, -2.7827095985412598, -1.5604958534240723, -1.6129176616668701, 0.12765942513942719, -0.5446857213973999, 0.283556193113327, 0.4769402742385864, -0.3579959273338318, -0.7186987996101379, -0.9161545038223267, -0.06046894192695618, 2.371286392211914, 3.7522828578948975, -0.903689444065094, -0.3478923439979553, 1.968910813331604, -1.2620888948440552, 0.21204209327697754, 0.8612043857574463, -0.53299480676651, -1.1089094877243042, 0.6155380010604858, -0.810754120349884, 0.9236339330673218, -1.9806240797042847, -0.06962233781814575, 0.10736491531133652, 1.412211298942566, -3.27412748336792, -3.3694467544555664, -0.8622068166732788, -0.21950693428516388, -1.327042818069458, -0.9585743546485901, -1.813928246498108, 1.8687688112258911, -1.0384477376937866, 0.5319466590881348, -0.884057879447937, 1.9406118392944336, 0.7545047998428345, 0.05433540791273117, -0.9669592976570129, 0.09471966326236725, 0.08744734525680542, -1.8603492975234985, -0.5098164677619934, 0.5630056262016296, 0.4765693247318268, 1.2969571352005005, 0.6350151896476746, -0.5781587362289429, -0.5357071161270142, 0.5327311158180237, 2.1891658306121826, 0.16838888823986053, 0.5572860836982727, -1.3704770803451538, -0.9482091665267944, 1.3965392112731934, 1.6865673065185547, -0.34185120463371277, -2.3522450923919678, -0.9601320028305054, 0.8113210797309875, -1.0666371583938599, -1.4745521545410156, 0.48376747965812683, 1.0037044286727905, 0.6513980627059937, -1.1697355508804321, -1.1914007663726807, 1.0629544258117676, 1.6752911806106567, -1.1853888034820557, -0.8661615252494812, -1.7623993158340454, -0.5621063709259033, -1.3512641191482544, 0.5352027416229248, 0.9803314208984375, -0.3585563004016876, -1.1690571308135986, 0.33045297861099243, 0.3120547831058502, 0.525091826915741, 0.08238593488931656, -3.0962259769439697, 2.0070998668670654, -0.7543833255767822, 5.077949523925781, -3.4314069747924805, -0.8932383060455322, -1.5748542547225952, 0.633651852607727, 1.3344777822494507, 2.1601431369781494, -1.3707584142684937, -0.4890862703323364, -0.8653626441955566, 0.6137763261795044, 0.8104864954948425, 0.12003844231367111, -0.2669845521450043, 0.04141180217266083, -1.6517550945281982, -2.488851308822632, 2.5180158615112305, 1.8211028575897217, 0.7838108539581299, -1.2969963550567627, 1.4300966262817383, -1.4822171926498413, 1.3637665510177612, 0.316622257232666, 0.6508204936981201, -1.2707505226135254, -0.38914236426353455, -0.365415096282959, 2.6155965328216553, -0.6081593036651611, 2.8140344619750977, 0.29509758949279785, -0.05262315273284912, 2.646303176879883, 1.149295449256897, -0.5956482291221619, -1.6273523569107056, -0.20004363358020782, -0.8404344916343689, 0.5794616937637329, -1.2190823554992676, -0.056923627853393555, 2.6199865341186523, 2.89044189453125, 1.438012957572937, 0.5068821907043457, 1.2731128931045532, -1.8837145566940308, 0.6237751245498657, 1.1453438997268677, 0.478423148393631, -1.1589497327804565, -0.9963605999946594, -0.789864182472229, -0.4929054379463196, -2.027801513671875, -0.6845274567604065, -0.7290470600128174, -2.161761999130249, -1.2463277578353882, 0.8920361995697021, -1.420589566230774, 0.10330422222614288, -0.32204172015190125, -0.13469356298446655, 1.3468135595321655, -0.3805772662162781, -1.6234017610549927, -0.441170334815979, -1.4622561931610107, 0.7848179340362549, 1.3708478212356567, -1.1630442142486572, 3.340362071990967, -1.3209296464920044, -1.5554574728012085, -1.4321616888046265, -1.1931891441345215, -1.9951210021972656, 1.5096691846847534, -0.06945481151342392, 1.031935453414917, 1.3707554340362549, -1.4628982543945312, -1.7073036432266235, 0.5668511986732483, 2.691403865814209, -1.7331784963607788, 0.10017518699169159, 0.1517324447631836, 0.2523137927055359, -1.4899711608886719, 0.9338474273681641, -0.7495294809341431, -0.9000928401947021, -2.4690258502960205], "sim": 0.6233462691307068, "freq": 357.0, "x_tsne": 121.60834503173828, "y_tsne": -56.91231155395508}, {"lex": "treatments", "type": "nb", "subreddit": "Coronavirus", "vec": [0.867694079875946, -1.1971808671951294, -1.0287539958953857, -2.3892428874969482, -2.1779799461364746, 1.7984665632247925, -0.39971745014190674, -0.3624894618988037, -0.3521096706390381, -0.162246435880661, 0.564594566822052, 1.9706764221191406, 3.6347508430480957, -1.0830512046813965, 3.5836901664733887, 1.675126314163208, -0.891732394695282, -0.6881005167961121, 4.616087913513184, 3.0793659687042236, -0.8231666088104248, 2.5594308376312256, -0.6957001090049744, -0.01420616079121828, 3.0948712825775146, 0.39908692240715027, 0.2268942892551422, 1.2797660827636719, 0.6059956550598145, -1.83047354221344, 0.6762993335723877, 0.9108745455741882, 1.0155500173568726, -2.1408655643463135, 2.0906717777252197, -0.26585230231285095, -1.214698314666748, -1.3945789337158203, -0.09085384011268616, -0.30987632274627686, -2.362563371658325, 2.6551034450531006, 0.17076033353805542, -0.008919975720345974, 2.1324048042297363, 1.4536147117614746, 3.0636675357818604, 2.606123924255371, -1.3684885501861572, -1.1379022598266602, -1.0104255676269531, -1.8326520919799805, -0.054123811423778534, 0.45347192883491516, -2.0217201709747314, -0.13529014587402344, 1.7004280090332031, 0.6060804724693298, -2.970262050628662, -1.6806262731552124, 2.6969497203826904, 3.3297717571258545, -0.10058388859033585, 3.061398506164551, 1.356451153755188, -0.6447202563285828, 0.9568770527839661, 1.2232067584991455, -0.21991609036922455, 0.09339269250631332, 0.08644472807645798, 0.09189137071371078, -0.48821815848350525, 0.6404548287391663, -1.8076056241989136, -2.8087046146392822, 0.9224230647087097, 1.3803166151046753, -1.3772937059402466, 3.350376844406128, 1.7489577531814575, -0.5683863759040833, -0.5971673727035522, -1.7003347873687744, 0.27206915616989136, -0.03729177638888359, -0.7695387601852417, -0.022135090082883835, 0.3229360580444336, 2.0664632320404053, -1.6629725694656372, -0.966914713382721, 1.763102650642395, 0.5777102112770081, -2.458838701248169, -1.8677276372909546, -0.03618517145514488, -0.6500411033630371, 1.9762364625930786, -1.320785641670227, 0.8847924470901489, 1.0493820905685425, -0.747455358505249, -1.0713828802108765, -1.9913734197616577, -1.8357656002044678, 3.1746766567230225, -1.838834285736084, 1.7452658414840698, -1.2087287902832031, 1.88083815574646, -3.4802193641662598, -0.8102242350578308, -1.4742419719696045, -0.36534351110458374, -0.32252296805381775, 0.17521966993808746, 0.5227681398391724, 0.3015177547931671, 0.23030097782611847, -0.8753652572631836, -2.408043146133423, -0.8819401860237122, 0.4243612587451935, 3.666478395462036, 2.8202481269836426, -0.00882804486900568, -0.5788583755493164, 3.3195009231567383, -1.6186164617538452, -1.455289363861084, -4.259621620178223, 1.0763252973556519, 0.13899661600589752, 1.3344837427139282, 0.765836238861084, 0.8486923575401306, -0.96441251039505, 0.5132253766059875, 0.3724774718284607, 2.0839061737060547, -3.9161016941070557, -1.5347483158111572, 1.0656872987747192, -0.6268205642700195, -2.6387598514556885, 0.5703206658363342, -2.3321614265441895, -0.22156359255313873, -2.774135112762451, 1.7500962018966675, 0.07998161017894745, 2.195474624633789, -0.9275897145271301, 0.1282796710729599, -3.369431972503662, 0.42020389437675476, 1.3882030248641968, -1.8336223363876343, -0.6426424980163574, -0.7214834690093994, 2.6268115043640137, 0.4040120542049408, -0.04038761928677559, 1.1471641063690186, 0.36583060026168823, 1.3119920492172241, 2.5534474849700928, -0.7327505350112915, 0.948146641254425, -2.023763418197632, -1.8280788660049438, 2.5740039348602295, 0.6284357905387878, -0.3622659742832184, -2.6243772506713867, -0.0429440401494503, 3.6291048526763916, 0.6504591107368469, -0.9901300072669983, 1.0816537141799927, 4.021434307098389, -2.0432534217834473, 1.015026330947876, -2.021129846572876, 2.4125418663024902, 3.7313499450683594, -2.214480400085449, 0.08357695490121841, -1.0140252113342285, 0.09174264967441559, -0.19897331297397614, 1.568945050239563, 0.5263227820396423, -2.9776697158813477, -1.821931004524231, -1.3470884561538696, 0.555826723575592, 0.38000068068504333, 2.7039430141448975, -4.07801628112793, 0.6857381463050842, -1.0841174125671387, 2.6461572647094727, -1.972184181213379, 1.7392946481704712, -2.4942123889923096, 1.5025144815444946, 1.0220140218734741, 2.653210163116455, -0.8978077173233032, 0.42071792483329773, -1.730116844177246, 1.00322425365448, -0.3313922584056854, 1.6255437135696411, -1.8106151819229126, -1.8160693645477295, -0.9646830558776855, -2.1815223693847656, 0.3404284119606018, 0.8306477665901184, -0.5138381719589233, -0.11366928368806839, 1.3119124174118042, -1.104225516319275, 2.1046793460845947, -1.0641624927520752, 1.5285030603408813, -0.7463433146476746, 0.46970200538635254, -0.1957523226737976, 1.757073163986206, -0.6347692608833313, 2.1128172874450684, 1.8486825227737427, 1.6611922979354858, 4.954899311065674, -0.26440951228141785, -0.8502965569496155, -2.6753406524658203, -0.33695855736732483, -1.5224902629852295, 1.67714524269104, -0.5420491695404053, -1.0031425952911377, 1.9825471639633179, 0.47812148928642273, 2.53462290763855, 0.38811591267585754, 1.3011109828948975, -0.781582236289978, 2.658200263977051, 1.9879460334777832, 1.1695146560668945, -0.5342603325843811, -2.6237199306488037, -0.3589104413986206, 0.8741264343261719, -2.598665952682495, -0.9123491048812866, 0.8397130966186523, -3.891892194747925, -1.7705049514770508, -0.04251417890191078, -0.5851815342903137, 1.6894458532333374, -1.4333339929580688, -0.39503100514411926, 1.5602145195007324, 2.1775574684143066, -1.365928292274475, -0.8236261606216431, 0.6112074255943298, 4.239962100982666, -0.5778967142105103, -1.1019847393035889, 2.12296724319458, -0.37861958146095276, -2.751666784286499, -1.5933088064193726, -2.5805065631866455, -1.20369553565979, 1.1596660614013672, 3.700486898422241, 0.3630910813808441, 0.7109442949295044, -0.09593109041452408, -1.9396392107009888, 1.0071675777435303, 2.312901735305786, -2.89508056640625, 0.7922200560569763, 1.549389123916626, 1.0322972536087036, -1.2866246700286865, -1.7805286645889282, -1.0239746570587158, -0.29887399077415466, -2.1377978324890137], "sim": 0.5975114703178406, "freq": 4737.0, "x_tsne": 146.61048889160156, "y_tsne": -10.219189643859863}, {"lex": "drugs", "type": "nb", "subreddit": "Coronavirus", "vec": [1.0349080562591553, -1.436851143836975, -1.1563054323196411, -1.8327937126159668, -3.545909881591797, 2.6289138793945312, 0.5989173054695129, 0.6435805559158325, 0.019896823912858963, -1.1599700450897217, -0.0605592280626297, 1.1342576742172241, 1.7477389574050903, -2.260235548019409, 2.708329916000366, -0.37172242999076843, 0.7409862279891968, -2.2035374641418457, 3.452054262161255, 0.12609218060970306, -1.0986013412475586, 1.2664062976837158, -0.23537775874137878, 0.4524771273136139, 5.69497013092041, -0.3504883050918579, 1.365252137184143, 0.6442295908927917, -0.12727874517440796, -2.7271127700805664, 2.193779945373535, -0.22794413566589355, 1.0940492153167725, -2.168121099472046, 0.5696686506271362, 0.40450480580329895, -1.647275686264038, -1.0364375114440918, -0.6213932633399963, -1.3563164472579956, -1.1831268072128296, 1.7458009719848633, -0.29409873485565186, 3.016380548477173, -0.2675192654132843, -0.6786110401153564, 2.835833787918091, 2.2125496864318848, 0.8597652912139893, -0.4322831928730011, 0.8698107004165649, 0.04205704480409622, 1.0999679565429688, 1.0616319179534912, -3.648656129837036, 0.2945629358291626, 0.4904204308986664, -0.2720734179019928, -0.20415939390659332, -2.8578720092773438, 0.6212491393089294, -0.4393545985221863, 0.7910842895507812, 0.7424942255020142, 1.2711797952651978, -1.6953099966049194, 1.7097158432006836, -1.5034148693084717, 0.04797002673149109, 0.37098753452301025, 1.0405361652374268, 0.8883078098297119, -1.3385810852050781, 0.7016667723655701, -1.4860302209854126, -2.9011354446411133, -1.340086817741394, 1.2338366508483887, 0.008314450271427631, 0.4773618280887604, 0.19613304734230042, -0.0755256935954094, -1.0993584394454956, 0.1306157112121582, -2.461345672607422, 0.9711771607398987, -1.2619471549987793, 1.2789157629013062, -0.347073495388031, 1.3022677898406982, -1.7895814180374146, -3.318228006362915, 1.0814251899719238, 1.4318528175354004, -2.187427043914795, -1.191755771636963, 1.5325847864151, -1.7185416221618652, -0.08051981031894684, -0.4296190142631531, -0.36021003127098083, 0.10570855438709259, 0.17694689333438873, -1.9790879487991333, -2.803561210632324, -1.5140341520309448, 1.1500691175460815, -1.0667105913162231, -0.8441610932350159, -0.5927361845970154, -0.7483732104301453, -0.9065988659858704, -2.2534921169281006, -3.157985210418701, -0.5801246762275696, -0.8908168077468872, 1.3212687969207764, -1.0877552032470703, -0.7528355121612549, 0.5198429822921753, -0.12275028228759766, -1.7499841451644897, 0.4691462814807892, -0.021923305466771126, 2.5713279247283936, 1.3714030981063843, 0.04800359904766083, -1.7844929695129395, 1.6150082349777222, -0.2484428435564041, -0.12922349572181702, -2.038146734237671, -1.7472271919250488, -0.5108626484870911, 1.379056692123413, -1.224777102470398, 0.9574376344680786, 0.45168331265449524, 2.5255632400512695, -0.7241536974906921, 2.582343816757202, -0.6726313233375549, -1.509913682937622, -0.8657016754150391, -0.4552951753139496, -1.3364636898040771, -0.7238157391548157, -1.7579189538955688, 1.7339115142822266, -3.0528783798217773, 0.12089981138706207, -1.7586586475372314, 2.3772854804992676, 0.8686076998710632, -0.19862967729568481, -0.9079335331916809, 0.20736199617385864, -0.3641432225704193, 0.14433756470680237, 0.13316014409065247, -1.1824716329574585, 0.6052121520042419, -0.9491876363754272, 0.11708234995603561, -1.5816842317581177, 2.2238121032714844, -0.24411891400814056, 2.460320472717285, 0.19195184111595154, 1.0209267139434814, -0.5793060660362244, -1.0418837070465088, 0.1442679613828659, 1.9560240507125854, 0.3203863501548767, -2.255155324935913, -1.342206597328186, 0.8816433548927307, -0.36443740129470825, -2.3358895778656006, 0.5177826285362244, 2.015202760696411, -0.32305166125297546, 0.15325811505317688, -0.9162801504135132, 1.2552809715270996, 2.7933802604675293, -0.39435428380966187, 0.14369113743305206, -0.8099445104598999, -0.05267683416604996, -0.6230928897857666, 1.5038539171218872, 0.6417486667633057, -0.04458270221948624, -1.3326746225357056, 0.3313920199871063, 1.534787893295288, 0.0015120103489607573, 0.7133731245994568, -3.384786605834961, -1.1220722198486328, -0.060714300721883774, 3.056612968444824, -1.4687408208847046, -1.604622721672058, -1.4430586099624634, 0.9170107245445251, 0.14050841331481934, 0.8972938060760498, -1.9203827381134033, -0.29331955313682556, -2.6798362731933594, 0.7532870769500732, 1.1087762117385864, -1.7652932405471802, -3.1074917316436768, -0.07684440165758133, -1.8137598037719727, -3.4078359603881836, 0.5368493795394897, 1.2437046766281128, 1.4795855283737183, 0.05206478014588356, 0.9162706732749939, -2.986241102218628, -0.384010374546051, -1.1087629795074463, 0.48433321714401245, 0.7979655265808105, 0.34954527020454407, 0.8290280699729919, -0.04139800742268562, -0.7211611866950989, 2.7434017658233643, -0.7414870262145996, 0.9708101153373718, 4.155966281890869, 1.2637569904327393, -0.3108619153499603, -3.5344204902648926, 0.23289932310581207, -3.1557631492614746, -1.731026291847229, -0.6287834644317627, -0.8450111746788025, 1.054693341255188, 0.5427617430686951, 2.488590717315674, -0.30912038683891296, 0.951362133026123, -3.42051362991333, 2.1678903102874756, -0.22440221905708313, 0.623171329498291, 0.30943840742111206, -1.0208075046539307, 0.6605772972106934, -0.7090437412261963, -2.5257790088653564, 0.6429562568664551, -0.5471622347831726, -2.5441863536834717, -3.50248384475708, 1.9481215476989746, -2.853024482727051, 1.8299232721328735, -2.101808786392212, 1.2467796802520752, 0.6797799468040466, 1.0220969915390015, 1.1418110132217407, -1.1808456182479858, -0.10139606148004532, 1.8398014307022095, 0.972235918045044, -1.7704672813415527, 2.5693202018737793, -1.010657787322998, -1.6560825109481812, -0.9311495423316956, -0.45696789026260376, -2.0582261085510254, 0.15508532524108887, 1.1944057941436768, 0.15716488659381866, 0.580990195274353, -1.6010727882385254, -0.07741769403219223, 1.809678077697754, 0.9096033573150635, -0.6830589771270752, 1.4005380868911743, 2.556363105773926, 2.0646121501922607, -1.3284965753555298, -1.5214900970458984, -0.010713226161897182, -1.6925196647644043, 1.5084787607192993], "sim": 0.5869288444519043, "freq": 13655.0, "x_tsne": 131.609130859375, "y_tsne": -20.842660903930664}, {"lex": "therapies", "type": "nb", "subreddit": "Coronavirus", "vec": [1.4403434991836548, 0.13577550649642944, 1.141663908958435, -2.3388068675994873, -0.7526288628578186, 1.4035362005233765, 0.5763753056526184, -0.04627785086631775, 0.8805223107337952, 0.10938144475221634, 0.07611580193042755, 1.1852995157241821, 3.1204349994659424, -1.9659409523010254, 2.358466148376465, -0.8780173659324646, 0.08025787770748138, -1.3831528425216675, 4.6566853523254395, 2.3415374755859375, 0.5894230008125305, 1.4541711807250977, -1.6698107719421387, 0.19443228840827942, 2.0983614921569824, 0.6501059532165527, 0.4063088595867157, -0.4294807016849518, -0.20739136636257172, -1.4185304641723633, 1.1896506547927856, 0.6460011601448059, 0.02205139771103859, -2.1847002506256104, 1.193197250366211, 0.0053064716048538685, -2.2081096172332764, -0.9243980050086975, -0.4436629116535187, -0.4389101564884186, -1.5071849822998047, 0.8909716010093689, 0.3974307179450989, 1.353944182395935, 2.0198938846588135, 0.9479495286941528, 1.9984421730041504, 1.3332948684692383, -0.6168928146362305, -0.5593637228012085, -0.09899202734231949, 0.1818453073501587, -0.9125462770462036, -0.38935691118240356, -2.521075963973999, 0.7284790277481079, 0.9149494171142578, -1.09958016872406, -1.1574726104736328, -1.8971360921859741, 2.174009084701538, 2.517564296722412, -0.9629002809524536, 3.1772286891937256, 1.3580254316329956, -0.7876845598220825, -0.22753509879112244, 0.8107059001922607, -0.7433796525001526, 0.7124216556549072, 0.057059887796640396, -0.8000375628471375, 0.058330800384283066, 1.3528406620025635, 0.1617230623960495, -0.9397374987602234, -0.5715606212615967, 0.5619966983795166, -1.7393392324447632, 1.6101785898208618, 1.3970485925674438, -0.6883306503295898, -1.5346317291259766, -2.3341212272644043, 0.1564493179321289, 0.571377158164978, -1.252198338508606, -1.1254773139953613, 0.05796759948134422, 2.09622859954834, -1.196956992149353, -1.5322219133377075, 2.5330426692962646, -0.49470362067222595, -1.9405549764633179, -1.6310144662857056, 0.013658491894602776, -1.7316319942474365, -0.4411104917526245, -1.2831878662109375, 0.9365225434303284, 0.18502458930015564, -0.6553399562835693, 0.7917775511741638, -2.22029185295105, -2.017392873764038, 2.7455101013183594, -0.37542441487312317, -0.719836413860321, -0.07108183205127716, 1.5551151037216187, -1.272291898727417, -0.9378142356872559, -1.927970051765442, -0.3049837350845337, 0.205555722117424, 0.3251027762889862, -0.052285533398389816, 0.9208162426948547, 0.7156649827957153, -0.15202772617340088, -2.682886838912964, -0.3299865424633026, -0.39410290122032166, 0.4473862648010254, 2.194751739501953, -1.0136487483978271, -0.8694440126419067, 0.8172364830970764, 0.2769342064857483, -0.6620717644691467, -1.9483531713485718, 1.231215476989746, -1.0034490823745728, 2.7589032649993896, -0.5861592292785645, 0.6354652047157288, -1.239309310913086, 0.4797891676425934, 0.8810058236122131, 1.0284545421600342, -2.0385401248931885, -0.37922075390815735, 0.014431004412472248, -0.23441874980926514, -1.2383430004119873, 0.25776803493499756, -1.1470783948898315, 0.24656030535697937, -1.9091744422912598, 1.6353682279586792, -0.04309682548046112, 1.409753441810608, -0.2801942229270935, 0.0046855974942445755, -0.9681347608566284, -0.6177929043769836, 1.1913601160049438, -0.04365655407309532, -0.8339989185333252, -1.7628289461135864, 0.2019134759902954, 0.19863192737102509, -0.8511842489242554, 0.41366010904312134, -0.8587482571601868, 0.13520969450473785, 0.9296482801437378, -1.4494960308074951, 1.0418201684951782, -2.247551918029785, -0.8878675699234009, 3.0107269287109375, 2.2240216732025146, 0.21056605875492096, -1.4641767740249634, -0.2651998996734619, 2.03320050239563, -0.2689058780670166, -2.256141185760498, 1.5179678201675415, 2.444889783859253, -0.4761950373649597, 1.1805554628372192, -1.8993488550186157, 1.8631551265716553, 2.588385820388794, -1.28095543384552, 0.4313100576400757, -1.313393235206604, -0.7768753170967102, -1.4184788465499878, -0.8715251088142395, 0.7454482913017273, -2.3013017177581787, -1.077450156211853, -0.9888742566108704, 1.1040246486663818, 0.6300694346427917, 1.2583764791488647, -2.7574336528778076, -0.0031437946017831564, -0.776531457901001, 2.171164035797119, -3.557140827178955, 2.2744009494781494, -3.468512773513794, 1.0848689079284668, 0.8554507493972778, 1.897104024887085, -1.1623969078063965, 0.6830852627754211, -1.9167406558990479, 0.30607160925865173, 0.36332985758781433, 1.8282649517059326, -2.711232900619507, -1.4274345636367798, -0.8121997714042664, -2.1056346893310547, 0.5994042158126831, 2.836944103240967, -0.7489945292472839, 1.1126607656478882, 0.749424397945404, -2.0307016372680664, 1.0974619388580322, 0.14103364944458008, 0.6735869646072388, -0.8332686424255371, -0.4510467052459717, 1.223992109298706, 2.5968148708343506, -0.4144606590270996, 1.8379502296447754, 1.3340955972671509, 0.2445499747991562, 4.700859069824219, -1.6814171075820923, -0.8252930641174316, -1.0375033617019653, -0.5873042941093445, 0.7209264636039734, -0.8888319730758667, -1.0016658306121826, -1.0538349151611328, 2.0431935787200928, 1.9136919975280762, 0.8331853151321411, 0.9755762219429016, 0.6669723987579346, -2.531811475753784, 2.433811664581299, 2.538930654525757, -0.30096501111984253, -0.8644974827766418, -1.5627014636993408, 0.07632716745138168, 1.226029872894287, -3.5145254135131836, -1.0196843147277832, -0.05195150151848793, -3.2725322246551514, -1.4642064571380615, 0.8137585520744324, -0.658990204334259, -0.42842990159988403, -0.8745883107185364, 0.6211894750595093, 0.968793511390686, 0.3485579490661621, -1.08024263381958, -0.838926374912262, 0.69004887342453, 3.029168128967285, 1.5188934803009033, -1.856899619102478, 1.101475715637207, -0.6586036086082458, -2.1623477935791016, -0.2928203046321869, -1.0047959089279175, -2.2657740116119385, 0.08101392537355423, 2.381690263748169, 1.0771605968475342, 1.1140516996383667, -1.460160493850708, -1.9271254539489746, 0.33755818009376526, 2.019493579864502, -0.6957733631134033, 1.115419626235962, 0.9089897871017456, 1.2337926626205444, -3.081860303878784, -0.29381561279296875, -0.41576191782951355, 1.5482689142227173, -1.385255217552185], "sim": 0.577518105506897, "freq": 425.0, "x_tsne": 124.17656707763672, "y_tsne": -6.465066909790039}, {"lex": "doses", "type": "nb", "subreddit": "Coronavirus", "vec": [0.11414876580238342, -1.7389497756958008, 1.3158471584320068, -3.0834357738494873, -1.4128785133361816, 1.8978694677352905, 2.247563600540161, -2.1353864669799805, -1.298112392425537, 1.6612111330032349, 1.78755521774292, 0.6920830607414246, 4.509463310241699, -2.2026896476745605, 3.028958320617676, 3.0450832843780518, -0.9170687794685364, -0.21054722368717194, -0.4485827386379242, -2.577094316482544, -2.2418487071990967, -2.6852476596832275, -3.4714834690093994, -0.18877151608467102, 0.88746178150177, -1.893484354019165, 1.1832184791564941, -0.6757444143295288, 1.3776761293411255, -0.027101164683699608, 1.0958162546157837, -2.301673412322998, 1.4399924278259277, -1.2633856534957886, 1.1887258291244507, 2.1824233531951904, -0.23483839631080627, -3.2586398124694824, -3.5039167404174805, -0.24048927426338196, -1.2500271797180176, 1.2937343120574951, 1.1030629873275757, 0.9731537103652954, 0.9957757592201233, -1.2917463779449463, 1.99351167678833, 0.0012531999964267015, -0.9827494025230408, -0.9467382431030273, 0.6249369978904724, 1.9640448093414307, -3.2039411067962646, -1.3421292304992676, -3.2272818088531494, -0.39214709401130676, 0.637016773223877, 2.084608793258667, 0.7525995969772339, -2.6471951007843018, 1.2851512432098389, 5.065957069396973, -0.5519033074378967, 1.6080255508422852, 1.0720243453979492, 0.04674839600920677, 4.205473899841309, 0.3664361536502838, -1.8427553176879883, 0.8493316769599915, -0.016294943168759346, -0.5306243896484375, -1.6019774675369263, -1.2732352018356323, -3.1119513511657715, -1.5110447406768799, 0.32029855251312256, -0.559402585029602, -0.07388097047805786, -0.5474201440811157, 1.2616668939590454, -1.0880240201950073, -2.0139243602752686, 2.654698610305786, -2.546016216278076, -1.2759554386138916, -0.5697916150093079, 1.474045753479004, 2.4955127239227295, -0.6534636616706848, -2.1468634605407715, 1.1079822778701782, 4.086485862731934, -0.14162307977676392, -1.3039906024932861, -2.271653890609741, -3.497288703918457, 1.4289298057556152, 1.1344037055969238, 0.7164363861083984, 1.460252285003662, 1.2041480541229248, -1.1377074718475342, -2.8337910175323486, -2.9555439949035645, -0.14101053774356842, 0.4529884457588196, -1.4925763607025146, 2.5397398471832275, -1.5612961053848267, -2.359673500061035, -5.137853145599365, 1.1370482444763184, -1.8718637228012085, -3.1194393634796143, -0.16795283555984497, -1.1228891611099243, 1.5988147258758545, 0.1175122782588005, 1.9661091566085815, -2.9284822940826416, -2.512225866317749, 1.3620686531066895, -0.8567964434623718, 0.7695643901824951, 3.9400768280029297, -1.328060507774353, 0.15124766528606415, -3.4599990844726562, -2.93506121635437, 0.9987207055091858, -2.1581435203552246, -6.591131687164307, -2.7991931438446045, 3.1332225799560547, 0.7880347371101379, -1.3367973566055298, -2.3528144359588623, -1.6104315519332886, 0.07573210448026657, 2.503343343734741, -1.447237491607666, -1.2598296403884888, 0.6738425493240356, -1.243195652961731, -0.6539095640182495, -1.9271137714385986, -3.385868549346924, 4.891745090484619, 0.5174644589424133, -0.8483030796051025, -0.12397030740976334, -0.24221190810203552, -0.5746955871582031, 0.1921224445104599, -2.9778103828430176, -1.4461190700531006, -0.5817331671714783, -2.1816797256469727, -0.7438856363296509, 2.302175283432007, -0.4379240870475769, -1.6796954870224, 2.151898145675659, 0.44844356179237366, 1.9789707660675049, 2.6055891513824463, 4.747476577758789, -0.1374470740556717, 4.497493267059326, 2.4247820377349854, -0.4928174912929535, -3.0827624797821045, -0.270975261926651, 1.2564949989318848, -1.6724804639816284, -1.8909765481948853, -2.8915514945983887, 0.5485605001449585, 2.1467864513397217, -1.233908772468567, 0.9235791563987732, -1.0947916507720947, -0.8364802002906799, -2.6976535320281982, 1.2829829454421997, 2.869680404663086, 0.8516870737075806, 0.9460229277610779, -3.253329277038574, -1.8059539794921875, -2.187575578689575, 0.12446384131908417, -0.10173153877258301, -1.6401995420455933, -0.5847532153129578, -2.2081353664398193, -2.8991942405700684, 1.0830824375152588, -0.6426516175270081, -2.1641900539398193, 2.2560460567474365, 1.065696358680725, 4.086465835571289, -1.6484040021896362, -0.8337824940681458, -1.048056721687317, 3.7265565395355225, -0.874297022819519, 0.7235066294670105, 1.4074876308441162, -1.809170126914978, -2.068061113357544, -0.6719164848327637, 1.6533522605895996, 1.4302839040756226, -1.6603349447250366, -4.604647159576416, 0.30837133526802063, -0.8145014047622681, -0.19984126091003418, 1.851979374885559, 2.8098864555358887, -2.3663735389709473, -2.082890272140503, -1.2675668001174927, 1.3678820133209229, 1.3111674785614014, 0.7141624093055725, -1.373030662536621, 2.3699393272399902, -2.561171770095825, -2.3691561222076416, 0.09898634999990463, 2.189610719680786, -2.73431658744812, 0.23844484984874725, 8.02460765838623, -2.8691201210021973, -0.918857753276825, 1.7658549547195435, 0.31312453746795654, 3.153726100921631, -3.066605806350708, 3.6918296813964844, 1.1067004203796387, -1.675342321395874, 1.6140700578689575, -1.1316765546798706, 0.06627505272626877, 1.2095484733581543, -1.2109516859054565, 1.8480890989303589, -1.3288286924362183, -0.8398939967155457, 3.5315895080566406, 2.951174736022949, 1.555242657661438, 0.7347521185874939, 1.3274163007736206, -0.389277845621109, -1.8296616077423096, -1.6766401529312134, -4.207309722900391, 2.7771430015563965, 5.262530326843262, 4.3744893074035645, -1.8595279455184937, -2.151522397994995, 1.4593619108200073, -1.5030829906463623, 1.8539563417434692, 1.0758692026138306, -1.7805391550064087, 2.750967502593994, -0.8092567920684814, -0.4613480567932129, 2.3704307079315186, -1.4665347337722778, -1.301279902458191, -2.213284492492676, -0.01726914383471012, -0.39479392766952515, 1.8408123254776, 4.516818046569824, -0.596947431564331, 1.4935719966888428, -3.071244955062866, -2.3750803470611572, -0.980470597743988, -2.16677188873291, -0.6483035087585449, -0.9518534541130066, 1.1253458261489868, 0.46683263778686523, 1.7886918783187866, 0.6192713379859924, -0.43941089510917664, -0.47495347261428833, -0.6288812160491943], "sim": 0.5690319538116455, "freq": 6558.0, "x_tsne": 138.1907958984375, "y_tsne": -26.12936782836914}, {"lex": "vaccins", "type": "nb", "subreddit": "Coronavirus", "vec": [0.45112890005111694, -0.48083582520484924, 0.2989461123943329, -0.20643901824951172, -0.11203623563051224, -0.3360675871372223, 0.31499966979026794, -0.16770470142364502, 0.012220486998558044, -0.4343145489692688, 1.1926183700561523, 0.09763755649328232, 0.9830116629600525, -0.47116100788116455, 0.9039700031280518, -0.03983175754547119, -0.021786175668239594, -0.5038357377052307, 0.21849624812602997, -0.17119210958480835, -0.2474118024110794, -0.21682575345039368, -0.17344269156455994, 0.5044865012168884, 0.8675420880317688, 0.3042524456977844, 0.4361391067504883, 0.2103138417005539, 0.11049234867095947, -0.29223155975341797, 0.46334919333457947, 0.12216354161500931, 0.18287205696105957, -0.19969254732131958, 0.4028235971927643, 0.14377149939537048, -0.04969484359025955, 0.13891158998012543, -0.2982000410556793, 0.3957940638065338, -0.6349371075630188, -0.003759369719773531, 0.010477956384420395, 0.3749942183494568, -0.5896084904670715, -0.7699839472770691, 0.006336904130876064, 0.14658349752426147, 0.04405883327126503, 0.16706547141075134, -0.5649663805961609, 0.0622650571167469, 0.1803646683692932, 0.5850220322608948, -0.22331778705120087, -0.09146609902381897, 0.42227259278297424, -0.09757647663354874, -0.2528330385684967, -1.0440247058868408, -0.08396100252866745, 0.552284836769104, -0.284981906414032, 0.17949795722961426, 0.35185736417770386, -0.15002809464931488, 0.31223952770233154, -0.12580648064613342, 0.23479843139648438, 0.2336924821138382, -0.23030662536621094, -0.5452398061752319, 0.5493649244308472, -0.1424693763256073, 0.5448353886604309, -0.34521475434303284, 0.2879030406475067, 0.37846583127975464, -0.37393680214881897, -0.013624938204884529, 0.055028147995471954, 0.03693440556526184, -0.5642291307449341, 0.31788015365600586, -0.7853365540504456, 0.7856072187423706, -0.5255690813064575, -0.14869332313537598, 0.23525816202163696, 0.1067642942070961, 0.22522807121276855, 0.10468007624149323, 0.7527023553848267, 0.5117101669311523, 0.01601395197212696, -0.36507439613342285, 0.017177486792206764, -0.21784338355064392, 0.6424437761306763, 0.1425463855266571, 0.5715777277946472, -0.6795000433921814, -0.35611552000045776, -0.5645057559013367, -0.9233971238136292, -0.27198100090026855, 0.11472605913877487, -0.3819495737552643, 0.40125179290771484, -0.04944188520312309, 0.7405706644058228, -0.44117864966392517, 0.19791589677333832, -0.46600157022476196, 0.049537286162376404, -0.2920916676521301, 0.827782928943634, 0.16265347599983215, -0.03194480016827583, 0.29582658410072327, -0.09192335605621338, -0.7430989146232605, -0.2759665548801422, -0.023057354614138603, 0.03082285262644291, 1.0861117839813232, -0.08742519468069077, 0.05315249040722847, 0.6770231127738953, -0.39571481943130493, 1.0302693843841553, 0.2636951208114624, -1.3010309934616089, -0.8032985329627991, -0.15572521090507507, -0.6224669814109802, -0.6678159236907959, -0.13726945221424103, 0.6834959983825684, 0.3631065785884857, 0.07751130312681198, -0.3452378809452057, 0.3534277081489563, 0.12901480495929718, 0.9337191581726074, -0.3715987801551819, 0.05563976615667343, -0.4719912111759186, 0.42088380455970764, -0.487039178609848, 0.4900152385234833, -0.04870956391096115, 0.640826404094696, 0.3192843198776245, -0.3429110646247864, 0.10878294706344604, -0.1292528212070465, 0.38155755400657654, -0.5739455223083496, 0.2381567656993866, 0.24084140360355377, -0.15889835357666016, -0.12534616887569427, 0.23417973518371582, -0.5207548141479492, 0.24604451656341553, 0.6449659466743469, 0.2782694101333618, -0.03216629475355148, -0.020873676985502243, -0.7222684025764465, -0.2966441810131073, 0.13976448774337769, 0.4478476643562317, 0.3395515978336334, 0.26625439524650574, -0.7831512093544006, -0.13905642926692963, 0.653933584690094, -0.21395769715309143, -0.09743897616863251, 0.5863206386566162, -0.26212218403816223, -0.22938412427902222, -0.42524391412734985, -0.2003021091222763, -0.14161799848079681, 0.007633552886545658, 0.078164242208004, -0.34373554587364197, -0.6607184410095215, -0.300421804189682, 0.4453065097332001, -0.29815196990966797, -0.6010119915008545, -0.4287137985229492, 0.26272979378700256, -0.20749057829380035, 0.27863484621047974, -0.16248255968093872, -0.4925879240036011, 0.08426225930452347, 0.28485339879989624, 0.2599518895149231, -0.39569714665412903, -0.21170549094676971, -0.0561034269630909, 0.5138221383094788, 0.6372893452644348, 0.44482913613319397, -0.23028889298439026, -0.5814846754074097, -0.8218657374382019, 0.02375231683254242, 0.30935701727867126, -0.10881552845239639, 0.13535209000110626, -0.8984888792037964, 0.21379846334457397, -0.28000888228416443, 0.5465611219406128, 0.22108086943626404, -0.4729137122631073, -0.16192841529846191, -0.015286468900740147, -0.2715167999267578, 0.27693477272987366, -0.13134765625, 0.30902618169784546, 0.18440143764019012, -0.06064952164888382, -0.19776488840579987, 0.10790561884641647, -0.1606992930173874, 0.6875961422920227, -0.14421546459197998, -0.03833356872200966, 0.47125720977783203, 0.023133737966418266, -0.25694817304611206, 0.17806756496429443, 0.07953521609306335, -0.25617268681526184, -0.7937518954277039, 0.36339324712753296, -0.6861556768417358, 0.18249095976352692, 0.39229583740234375, 0.4988270401954651, 0.3013347089290619, 0.42895570397377014, -0.3485693037509918, 1.1081194877624512, 0.21547916531562805, 0.6647146344184875, 0.4767245054244995, -0.19740889966487885, -0.43122023344039917, -0.1326739490032196, -0.3394167125225067, 0.12678295373916626, -0.015228133648633957, 0.06646568328142166, -0.5192765593528748, 0.791115939617157, 0.4148336350917816, 0.14433208107948303, 0.4789702594280243, 0.19236576557159424, 0.3636154532432556, -0.07222563773393631, -0.11859719455242157, -0.09817633032798767, -0.01840023882687092, 0.1909162849187851, -0.010569058358669281, 0.13606205582618713, 0.4509384334087372, 0.07670508325099945, -0.30524447560310364, 0.17518268525600433, 0.6720443367958069, -0.31278935074806213, -0.15705536305904388, 0.05578318238258362, 0.011896129697561264, 0.22617723047733307, -0.4272322356700897, -0.008768579922616482, 0.07088032364845276, 0.19464430212974548, 0.1574525088071823, 0.7400631904602051, -0.03607902675867081, -0.5137300491333008, 0.7505204081535339, 0.054146990180015564, 0.6121009588241577, 0.06723306328058243, 0.09011543542146683], "sim": 0.5474942326545715, "freq": 29.0, "x_tsne": -25.22141456604004, "y_tsne": 65.69273376464844}, {"lex": "trials", "type": "nb", "subreddit": "Coronavirus", "vec": [-1.135069489479065, -1.1981315612792969, -3.253303050994873, -1.2904633283615112, 1.1460299491882324, 5.149393558502197, -1.1048129796981812, -1.855746865272522, -0.4513017237186432, -0.3064834773540497, 0.35525912046432495, 3.17315673828125, 3.611186981201172, -1.2537699937820435, 2.979196310043335, 1.672369122505188, -0.002093543531373143, -4.616672992706299, 1.127059817314148, 0.5472252368927002, -1.9323139190673828, 1.7953532934188843, -1.9249777793884277, -3.768315076828003, -1.9525549411773682, -1.2029122114181519, -1.9265832901000977, 0.3363676965236664, -2.3968076705932617, -0.3659287095069885, 1.8408126831054688, -2.9128026962280273, 1.1652776002883911, -1.1738965511322021, 1.6487776041030884, 0.10536682605743408, 0.3778116703033447, 0.42743489146232605, -4.113850116729736, 1.1439905166625977, -2.3262479305267334, -0.15591219067573547, 0.7620921730995178, 1.8212732076644897, 2.182258367538452, 0.35136058926582336, 0.8010126948356628, 1.4943621158599854, -1.3564375638961792, -2.653384208679199, -0.9713109135627747, 1.544434666633606, -0.6819599270820618, 2.227421283721924, -0.5851476788520813, -0.20864425599575043, -0.3843337297439575, -0.34843599796295166, -2.131099224090576, -2.895566940307617, 0.35899457335472107, 2.137410879135132, -0.7068741321563721, -0.24287866055965424, 0.11616434156894684, 0.5164342522621155, 1.4540677070617676, -0.5960162281990051, -1.812534213066101, 1.8994667530059814, -0.7180501222610474, -0.5431597232818604, -5.002516746520996, 1.6444261074066162, -1.3356292247772217, -1.666471004486084, 3.7584128379821777, 2.3587193489074707, -0.36092081665992737, 2.994318723678589, 1.4574106931686401, -1.3215405941009521, -2.721160650253296, -0.9947741627693176, -1.891527533531189, -1.1850908994674683, -1.1087968349456787, 1.01249361038208, 2.4405858516693115, 1.4576411247253418, 0.09495575726032257, -1.5798851251602173, 2.3112666606903076, -0.5168903470039368, -0.5593670010566711, -1.2659059762954712, -1.8460830450057983, 4.341877460479736, -0.4709974229335785, -1.0380147695541382, 1.2541524171829224, 2.973022699356079, 0.36567816138267517, -0.848358154296875, 0.19730274379253387, -1.2778674364089966, -0.31113484501838684, -0.18602368235588074, -0.40871039032936096, 0.9477878212928772, -0.17624850571155548, -2.653575897216797, -2.3411803245544434, -3.5978565216064453, -1.5504096746444702, -0.34403955936431885, -1.646235704421997, -0.09341899305582047, -1.4404531717300415, 1.0138987302780151, -4.188938140869141, 1.4941717386245728, -2.150233268737793, 1.4949959516525269, 1.3403725624084473, 4.44597864151001, 0.12396173924207687, 0.06659969687461853, 1.918447732925415, -1.1316285133361816, -0.3237863779067993, -2.726231098175049, -3.787830352783203, -0.7202432155609131, 3.1980950832366943, -1.1129902601242065, -2.2788443565368652, -0.8435361385345459, -0.6733729839324951, 0.12171496450901031, 3.0856082439422607, -2.312014579772949, 0.2168198525905609, 0.0411221943795681, 3.0353341102600098, -1.044844150543213, -0.3353215754032135, 0.5665143728256226, 4.143985271453857, -0.85100919008255, 0.5175347328186035, 1.7002317905426025, 0.1400861144065857, -3.585210084915161, 0.29586881399154663, -1.7206209897994995, -3.057481288909912, 1.653817892074585, -1.9396501779556274, 0.6253990530967712, -1.5334107875823975, 1.9769964218139648, 1.0689811706542969, -1.4991121292114258, 0.3738931715488434, 1.5501242876052856, 2.6724538803100586, 2.2123184204101562, 0.4326508343219757, 2.4491446018218994, 1.9308379888534546, 0.34927898645401, -1.570797324180603, 4.443512439727783, -2.4565157890319824, -1.7849115133285522, -1.8726240396499634, -0.26867491006851196, 0.09772581607103348, -2.29970383644104, -2.603564739227295, 0.7579715251922607, -0.6902057528495789, 0.007710359524935484, -4.837513446807861, 0.2208976298570633, 2.42592716217041, -3.078137159347534, 0.6261970400810242, -2.0586585998535156, -0.5706366300582886, 1.452613115310669, -1.8914505243301392, 0.5902856588363647, 0.8734390735626221, -3.9083235263824463, -0.630145788192749, -0.7015259861946106, 1.0967659950256348, 0.5153185725212097, -3.110663652420044, 0.714951753616333, -1.69231116771698, 1.5114518404006958, -1.685316801071167, 0.9201820492744446, -0.9445181488990784, 0.1724107563495636, 1.0353986024856567, -1.3466943502426147, 0.8298680782318115, 0.24418525397777557, 0.17536458373069763, 0.039820484817028046, 1.6926122903823853, 1.9972063302993774, -0.5189582109451294, -2.222628355026245, 0.19986386597156525, -0.33444809913635254, 0.7837085127830505, 2.9640846252441406, 0.01357631292194128, -3.2338459491729736, -1.2643789052963257, -1.0166524648666382, -0.12061326205730438, 0.9918028116226196, 0.5240816473960876, -0.8053844571113586, 0.782402753829956, -0.8713477253913879, -2.9493677616119385, 2.0278854370117188, 2.944795608520508, -0.8431453108787537, 0.597793459892273, 5.348711013793945, -2.328298330307007, 0.1718883216381073, -3.8287672996520996, -2.066779613494873, -0.8049290180206299, -2.9832346439361572, -2.118929624557495, -2.7763707637786865, 1.5329506397247314, -1.2954298257827759, -0.23867399990558624, 0.0476338267326355, 2.0642592906951904, -1.762974739074707, 3.91593337059021, -0.4138292372226715, -1.987152338027954, 1.1033190488815308, 4.055490016937256, -0.412967324256897, -1.706491231918335, -0.5039092898368835, -2.8725411891937256, -0.7688083052635193, -3.9506986141204834, -1.1595226526260376, 1.7617548704147339, -2.941566228866577, -0.5603285431861877, 0.6216719150543213, 1.0572867393493652, 1.916578769683838, -0.9210531115531921, -0.6015377640724182, 1.074637770652771, 0.15721789002418518, 2.423208713531494, 3.897495746612549, -0.15808795392513275, 0.588422954082489, -0.4165777862071991, -1.7436858415603638, -1.7553354501724243, -1.6996253728866577, -1.8632251024246216, -1.677856683731079, 0.23535603284835815, -1.0956711769104004, 2.596604585647583, -0.6575505137443542, 0.7468263506889343, 3.906013011932373, 1.034989356994629, -1.7498120069503784, 0.5882886052131653, 0.6217907071113586, -0.1413017064332962, -2.0882527828216553, -2.1165354251861572, 0.8185383677482605, -0.24926111102104187, -1.2183979749679565], "sim": 0.5451608300209045, "freq": 10095.0, "x_tsne": 130.67262268066406, "y_tsne": 22.02230453491211}, {"lex": "strains", "type": "nb", "subreddit": "Coronavirus", "vec": [3.672264814376831, 1.3105580806732178, 0.5657199025154114, -0.7929362058639526, 1.9042809009552002, -0.9151714444160461, 3.1579439640045166, 0.34773901104927063, -2.8330914974212646, -2.4600741863250732, 3.572629690170288, -1.9252785444259644, 2.9313786029815674, 0.595905065536499, -0.36492449045181274, 0.055739134550094604, -0.5195341110229492, -1.7868707180023193, -0.9681570529937744, 0.8739567995071411, 0.5900635719299316, -1.182973861694336, 0.028506511822342873, -1.8451364040374756, 3.6171581745147705, 1.4822487831115723, -0.2176070660352707, -0.5747644305229187, -0.9438624382019043, -1.8020472526550293, -0.653123676776886, -1.9940788745880127, -1.4964572191238403, -2.9479730129241943, 3.575803279876709, 0.5075213313102722, 0.005298810079693794, -2.7463314533233643, -2.382113456726074, -0.31185057759284973, -1.1922674179077148, -0.10285284370183945, 1.9927419424057007, -2.1073105335235596, 2.1676430702209473, 1.0289413928985596, 0.3547344505786896, 1.2575167417526245, -2.3042149543762207, -1.1369097232818604, -1.1958072185516357, -2.6582839488983154, -0.38235387206077576, -1.2978720664978027, -1.6177915334701538, 2.8665051460266113, 0.5610338449478149, -1.1941516399383545, 0.8399661183357239, -4.171733379364014, 0.37337350845336914, -0.22550208866596222, 0.12364520877599716, -1.5185452699661255, -1.9191900491714478, 0.3897557854652405, 1.3069862127304077, -1.5380308628082275, -3.0218987464904785, -1.2510162591934204, 3.78544545173645, -1.7725337743759155, 1.254286289215088, -0.822555661201477, -1.0044859647750854, -3.411771774291992, 1.959065556526184, 0.686438262462616, -2.3850715160369873, 4.830039978027344, 1.7908521890640259, -1.4285167455673218, -4.470229625701904, -0.5261987447738647, -2.95804762840271, -0.8519943356513977, -2.0153610706329346, -1.194679856300354, -0.7325360774993896, 0.39720624685287476, -2.7298107147216797, -0.1978009045124054, 0.9150559902191162, -0.6521133780479431, -0.38932666182518005, -1.6468188762664795, -3.067776679992676, -0.6179455518722534, 0.5304916501045227, -2.045626640319824, -0.4446919560432434, 0.2578675448894501, -0.29787948727607727, 0.7419628500938416, -1.3920732736587524, 0.2519581615924835, -1.7967791557312012, -1.9804350137710571, 6.776974201202393, -1.503644347190857, 2.7587289810180664, -2.856121063232422, -1.1217905282974243, -1.7957442998886108, -2.1249122619628906, 1.3112678527832031, -1.4408369064331055, -0.4386660158634186, 2.7724337577819824, 1.1694320440292358, -2.3914434909820557, -1.5131694078445435, -1.3751524686813354, -4.2201995849609375, 2.8743975162506104, 4.8480329513549805, -0.5460302829742432, 2.1560866832733154, -1.725324273109436, 0.39122235774993896, -0.2633429765701294, -2.705165386199951, -2.6764960289001465, -3.074864387512207, -0.7314573526382446, 0.975354015827179, 0.9376470446586609, -2.150017261505127, 2.519580841064453, -1.1180293560028076, 2.0071957111358643, -4.349266529083252, -2.3089654445648193, 0.304196298122406, -1.477500557899475, -0.6051223874092102, -0.5726848840713501, -5.144372940063477, 1.6614903211593628, -1.066066026687622, 0.6496915817260742, 0.655283510684967, 1.4546207189559937, -1.111870288848877, -0.33655402064323425, -0.8493330478668213, 3.139044761657715, 0.020971301943063736, -0.5184305310249329, -1.0720219612121582, -1.118931770324707, 1.703600287437439, -0.2578689157962799, 2.6948399543762207, 0.8384521007537842, 0.397522509098053, -1.4476453065872192, 0.9129857420921326, 1.1572085618972778, -0.20837430655956268, -0.8011302351951599, 1.442334532737732, -1.2704607248306274, 0.3236457109451294, 0.8145957589149475, -1.8669307231903076, -1.710492491722107, 0.4799482226371765, 1.9057471752166748, -1.2292895317077637, -2.2266640663146973, 2.2841904163360596, 2.0523667335510254, 0.6445507407188416, -3.606092691421509, -0.1990661472082138, 1.2415118217468262, -3.768864393234253, 0.39188629388809204, -2.5785257816314697, -0.5407175421714783, -1.2149100303649902, -0.612453043460846, 2.58341908454895, -0.9538540840148926, -2.400498867034912, 2.517270803451538, -0.005640062037855387, 0.3860672414302826, -0.6086292862892151, -0.8819834589958191, 1.0895637273788452, -2.1160075664520264, 2.404268264770508, -3.8986470699310303, 1.127197265625, -3.188709020614624, 2.6405296325683594, 0.3781428635120392, 0.5495492219924927, -1.1941581964492798, -0.39626720547676086, -0.4779132902622223, 1.902584433555603, 0.6215150952339172, 0.5858186483383179, -2.776843786239624, -2.038543701171875, 1.320591926574707, -0.10093903541564941, 1.2554644346237183, 1.4523975849151611, 1.435050129890442, -0.39670848846435547, -1.2303047180175781, 2.012470006942749, 1.4858626127243042, -0.19030416011810303, 1.427107572555542, -3.8666627407073975, -0.07418614625930786, -0.5226868391036987, -3.7280707359313965, 0.11904489994049072, 1.2735695838928223, -1.0628770589828491, -0.6711476445198059, 2.386857509613037, -1.4437024593353271, -0.6144860982894897, -1.2353827953338623, -2.8503811359405518, 0.28099504113197327, -2.385213851928711, -1.0314558744430542, -1.6075810194015503, 5.011438369750977, 0.903427243232727, 0.3674880266189575, 2.086087226867676, -0.0439440943300724, -1.4097129106521606, 3.5628268718719482, 1.4023269414901733, -0.22003844380378723, -0.4163091778755188, -0.42407745122909546, -2.19913649559021, -0.026126950979232788, -1.216220498085022, 1.64553964138031, -1.252960205078125, -2.238987684249878, -0.5806149840354919, 2.745189666748047, -0.06872029602527618, 2.834451675415039, 0.31484469771385193, -1.3733159303665161, 3.103827953338623, 0.4726307690143585, -0.0084627540782094, -2.1531894207000732, 0.23110264539718628, 0.9615007042884827, 2.3466246128082275, -1.280018925666809, 7.120027542114258, -3.064709186553955, 0.9984781742095947, -0.7630044221878052, 0.9218611121177673, -4.325984954833984, 1.771824836730957, 0.7747730612754822, -1.2369580268859863, 0.5990558862686157, -3.7865684032440186, -1.6071206331253052, 0.2256229668855667, 1.1688554286956787, 0.8246520161628723, 3.8843276500701904, -0.4511962831020355, 0.6280716061592102, 0.2325863242149353, 1.043945074081421, 0.05906497314572334, -2.0763659477233887, 0.07133735716342926], "sim": 0.5384674668312073, "freq": 3875.0, "x_tsne": 168.93649291992188, "y_tsne": 30.01032829284668}, {"lex": "vaccinations", "type": "nb", "subreddit": "conspiracy", "vec": [-0.9648794531822205, 0.5519004464149475, 0.6992305517196655, 0.03338951617479324, 1.0731843709945679, -0.12930205464363098, -1.0387333631515503, -0.9641603231430054, 1.6864888668060303, 1.2956066131591797, 2.557135820388794, 1.3338631391525269, 2.1533143520355225, -0.6863566637039185, 3.1992247104644775, 3.443852663040161, -0.33309870958328247, -1.7646186351776123, 0.3809446692466736, 1.2763950824737549, -0.32661518454551697, 0.9829246401786804, -1.5237911939620972, 0.4915373921394348, 0.7555075287818909, 1.9683663845062256, 0.33573874831199646, 1.2788009643554688, -0.48924726247787476, 0.6265057921409607, 0.39862948656082153, -1.2421410083770752, 1.2784167528152466, -2.5867421627044678, -0.6338188648223877, 0.7342889904975891, -1.1662707328796387, -1.3285934925079346, -2.3904519081115723, 0.31902599334716797, -1.7538204193115234, 0.4872291684150696, -2.0487425327301025, -0.2092670351266861, 0.763412594795227, -0.5187900066375732, 1.12659752368927, 0.6969115734100342, 2.2170369625091553, -1.8940916061401367, -0.16775518655776978, -0.3743438720703125, -1.9124425649642944, 0.15641902387142181, -1.9736303091049194, -1.012939214706421, 1.117397427558899, -0.6537942886352539, 1.1559239625930786, -1.0502296686172485, -1.2817072868347168, 0.8993198871612549, -1.8543014526367188, 1.6142915487289429, 0.9505153894424438, 1.5835508108139038, -0.6807636618614197, -0.46285659074783325, -0.5514408946037292, 1.5523817539215088, -0.13886941969394684, -1.9014936685562134, 0.2621396481990814, 1.5091004371643066, -1.9003609418869019, -2.9989004135131836, -0.5868408679962158, -0.05512387678027153, -0.1628035008907318, -1.3681490421295166, 1.726978063583374, 0.7020398378372192, 0.5378199219703674, -0.10940232872962952, -1.700260043144226, 0.12689770758152008, -1.7391904592514038, -0.16911813616752625, -0.8042429089546204, 1.052955150604248, -0.10445795953273773, 0.13712140917778015, 3.6144661903381348, 2.9260809421539307, -1.8201558589935303, -1.9275931119918823, 0.04971561208367348, 0.47512170672416687, 1.454970359802246, -0.6638615131378174, 0.6602470278739929, -2.0453310012817383, -0.46580106019973755, -1.7328789234161377, -1.4516130685806274, -3.385796070098877, 2.0439612865448, -0.6833614706993103, 1.2027866840362549, 1.4318546056747437, -0.7954850196838379, -2.754004716873169, -1.5241864919662476, -0.25572073459625244, -1.2930244207382202, -1.2265530824661255, -1.0597519874572754, 1.968772292137146, 0.26830020546913147, -0.4140014350414276, -0.012716934084892273, -1.3760968446731567, 1.0197513103485107, 0.5616772770881653, 0.5261010527610779, 3.400453805923462, 0.5887894630432129, -1.5633666515350342, -0.5817058682441711, 0.8972880244255066, 0.055307500064373016, 1.476514220237732, 0.49309346079826355, -1.7206079959869385, 2.038147449493408, -1.0101679563522339, -0.5128566026687622, -1.5255581140518188, 0.1539943516254425, -0.4772188663482666, 1.1471718549728394, -2.0397346019744873, -0.1754603236913681, 1.016113042831421, 0.3691655993461609, -3.2130136489868164, 1.5553122758865356, -0.5429325699806213, -0.3669019937515259, -1.213089942932129, 0.12098630517721176, 0.29109856486320496, -0.9778830409049988, -2.334012985229492, -1.2446571588516235, -1.1523855924606323, -2.1427817344665527, -1.5793529748916626, -3.0023880004882812, 0.9055604934692383, 0.20895978808403015, 1.6904401779174805, -0.5529672503471375, 0.7768228054046631, -0.3250086307525635, -0.38872238993644714, 0.0735606700181961, 2.22019100189209, -2.0861005783081055, 0.29247143864631653, -1.8076974153518677, 0.6472780704498291, 1.0542041063308716, -0.866675853729248, 1.5122727155685425, 0.5455989837646484, -1.523876428604126, -1.0039376020431519, 0.9053989052772522, -0.2677067816257477, 0.26066306233406067, 2.017270088195801, -1.4220097064971924, -2.136240005493164, -1.788284182548523, -0.29968637228012085, 1.8625235557556152, -0.7745571732521057, -1.2873475551605225, 0.7967487573623657, -1.012818455696106, -0.08094178140163422, 0.9218556880950928, -0.2869543135166168, 0.45394909381866455, -1.8995182514190674, -3.8175413608551025, 0.33766165375709534, 0.6969953775405884, 1.3860161304473877, 0.32063913345336914, -0.6602721214294434, 0.0038052499294281006, 1.7984436750411987, -3.435570240020752, 0.20257316529750824, -0.6148828268051147, -0.46767085790634155, -0.00392487645149231, -0.5692950487136841, 1.2763972282409668, -0.5631423592567444, -0.3520039916038513, 0.834486186504364, 1.8300420045852661, 1.2233164310455322, -1.1557121276855469, -1.0656663179397583, 0.906955361366272, -1.011021614074707, -0.1452912539243698, 0.40366578102111816, 1.3297107219696045, 0.35125279426574707, -2.3848657608032227, -1.2756165266036987, -0.2002386897802353, 1.5374321937561035, 2.256368398666382, -0.30049577355384827, 1.1712509393692017, -0.47878363728523254, -1.088202714920044, 1.3401376008987427, 2.848376512527466, 1.2930867671966553, 0.6355887055397034, 2.1275761127471924, -2.213099241256714, 1.8741817474365234, 0.26389646530151367, -0.5565497875213623, -1.4279892444610596, -1.877345323562622, 0.12764231860637665, 1.570465087890625, 1.6610385179519653, -1.3340270519256592, 0.777273952960968, 1.2697081565856934, 0.9522150754928589, -1.0345590114593506, 0.6091811060905457, 2.3074123859405518, 1.676416039466858, 1.0715141296386719, 0.022606711834669113, -0.1384371668100357, -1.8906339406967163, 0.838049054145813, 0.08824830502271652, 1.0533087253570557, 0.20379656553268433, -2.1849629878997803, 0.3087424337863922, 0.05278124660253525, -0.6396796703338623, -1.710194706916809, -3.448741912841797, 1.606377363204956, 0.4482506215572357, -3.586545944213867, -2.5366690158843994, -1.9389034509658813, 0.10807651281356812, -1.3274853229522705, -1.1639025211334229, 2.9290311336517334, -1.6954964399337769, 0.17091697454452515, -0.4987092912197113, -0.708203136920929, 0.29982060194015503, 3.152026414871216, 2.4647772312164307, -1.1176865100860596, 0.8906473517417908, -1.3733028173446655, -1.4760929346084595, 0.7835376262664795, 0.8069968819618225, 0.8597586750984192, 1.2545310258865356, 0.8612785339355469, 0.7966449856758118, 1.7156583070755005, -0.6119146347045898, -2.409634590148926, -0.7670199275016785, 0.00829855352640152], "sim": 0.8027861714363098, "freq": 3624.0, "x_tsne": 86.88040161132812, "y_tsne": -48.074302673339844}, {"lex": "vaccine", "type": "nb", "subreddit": "conspiracy", "vec": [-0.13505546748638153, -0.4962478578090668, 1.0700243711471558, -1.1448382139205933, -0.4675479829311371, -1.1753976345062256, 0.02165183238685131, 0.03497881069779396, 0.9741021990776062, 2.396921157836914, 1.7438607215881348, 1.6220777034759521, 5.7914719581604, -3.319284200668335, 4.017467021942139, 2.7855961322784424, 1.265132188796997, -1.6194740533828735, 0.4926425516605377, 1.994134545326233, -0.27125778794288635, -0.2783501148223877, -1.037587285041809, 2.6097772121429443, 3.0879812240600586, 1.9761043787002563, 0.867452085018158, 1.5376795530319214, 1.9617538452148438, 0.02911691926419735, -0.7649770379066467, -2.734342098236084, 0.7223820686340332, -2.3728294372558594, 0.9765070080757141, 1.9285626411437988, -2.9979984760284424, 0.282390296459198, -3.4393808841705322, 2.1818647384643555, -0.09361813962459564, 0.7329732179641724, -1.639968991279602, 2.0304627418518066, 1.3990942239761353, -0.8377153873443604, 1.9386204481124878, -0.1416337937116623, 1.254503607749939, 0.09265083074569702, 0.05054997652769089, -2.9886090755462646, -0.25121206045150757, -1.4124000072479248, -2.6755502223968506, -0.8514336943626404, 2.2114417552948, -2.1516001224517822, -1.1852006912231445, -1.7722339630126953, -0.8427306413650513, 3.4141151905059814, -1.7379522323608398, 2.5718889236450195, 1.7892216444015503, -2.393775463104248, -3.744952917098999, 0.9465858936309814, -0.769557535648346, 2.957711696624756, 1.7340322732925415, -1.325465440750122, 0.2779318392276764, 1.2463127374649048, -1.6226999759674072, -5.713033676147461, -0.16391459107398987, 0.41520747542381287, -2.015740394592285, -0.6807302832603455, 0.21907946467399597, -0.8880208134651184, -0.1893460750579834, 2.3378779888153076, -2.5727617740631104, 0.9490629434585571, -3.4308829307556152, -2.83046293258667, -0.9277099967002869, 2.9772305488586426, 1.4612226486206055, 0.17314453423023224, 2.1515958309173584, 2.424562692642212, -1.1862350702285767, -2.4855260848999023, -1.0359703302383423, 2.2300431728363037, 0.667485773563385, 0.9187188148498535, 0.615260899066925, -1.8288935422897339, 0.7774680256843567, -0.9745217561721802, -2.439472198486328, -3.6577839851379395, 1.9628777503967285, -3.1647095680236816, 0.533500611782074, 3.226834297180176, -0.5501553416252136, -4.426014423370361, -1.8310141563415527, 0.09486941993236542, 0.03918193653225899, 0.6457732915878296, 1.8629040718078613, -1.3568878173828125, 1.3514050245285034, -1.5639533996582031, -2.0451719760894775, -2.39154052734375, 1.5484323501586914, 1.5821815729141235, -0.8361411094665527, 3.4522716999053955, 1.0900962352752686, -3.292489767074585, 0.0029318509623408318, 1.3200041055679321, 1.095870852470398, 2.1099205017089844, 0.5887566208839417, -1.694534182548523, 1.5924863815307617, -0.2457830011844635, -2.4437546730041504, -2.2016401290893555, -0.2947317659854889, -0.24948939681053162, 1.5695761442184448, -3.15120530128479, -1.0676376819610596, 2.8061349391937256, 1.2586313486099243, -3.870744466781616, 0.3664366900920868, -0.5381445288658142, 1.0464134216308594, 2.0318963527679443, 0.008859957568347454, 0.061030130833387375, -2.2724812030792236, -1.777603268623352, -1.2125788927078247, -2.342992067337036, -1.9122693538665771, 2.0761771202087402, -4.292980194091797, 3.9687507152557373, 0.9351462721824646, 1.893278956413269, -0.8649920225143433, 2.3779401779174805, -0.422493577003479, -2.256680488586426, 0.35545238852500916, 1.7531896829605103, 0.8458455204963684, 1.5289610624313354, -2.0025572776794434, 0.4037896692752838, -0.2622295320034027, 0.04539279267191887, -0.4408600330352783, -0.28279542922973633, -2.0390756130218506, -0.4594862163066864, -2.0235435962677, 2.1349873542785645, -1.5383598804473877, 2.2794911861419678, 1.6892201900482178, -1.9956873655319214, -0.2458941638469696, -1.3098783493041992, 2.197845220565796, 0.6773266792297363, -1.0236916542053223, -1.2735515832901, -2.0537309646606445, 0.7331258058547974, -0.05685766041278839, 0.6537879705429077, 1.0652828216552734, -0.9353467226028442, -1.823508858680725, -1.2779461145401, 0.10443323105573654, 0.02882813662290573, -2.2402584552764893, 1.3734925985336304, 0.7182552218437195, 3.3010175228118896, -3.5875465869903564, 1.5733144283294678, -2.6840860843658447, -0.6845090389251709, 0.993863582611084, -1.0445992946624756, 2.641907215118408, -1.2470955848693848, -2.138683319091797, 1.3798637390136719, 2.2355709075927734, -1.0938656330108643, -2.1754884719848633, -1.8130425214767456, 1.2484130859375, -3.105414867401123, -2.1363306045532227, 0.15171197056770325, 1.3586766719818115, -1.6693438291549683, -3.2387261390686035, -0.49106279015541077, -1.6814366579055786, 1.0802372694015503, 0.19841276109218597, -0.005361510906368494, 0.8115330338478088, -0.437101811170578, -1.5339388847351074, -0.591639518737793, 4.584711074829102, 1.4644540548324585, -1.4270827770233154, 3.1424314975738525, -2.1801505088806152, 1.2948411703109741, 1.086219072341919, -1.558846354484558, -1.4098118543624878, -1.9779261350631714, 0.10701891034841537, 0.7045993804931641, 2.106189012527466, 3.535844564437866, -1.1574006080627441, 0.7823616862297058, 1.0481640100479126, -2.722517490386963, 1.6280606985092163, 2.9302847385406494, 0.18869027495384216, 2.032474994659424, -0.7820578217506409, 0.23599912226200104, -2.3668251037597656, 2.1576812267303467, -0.33990105986595154, 0.3705199062824249, -0.1969175785779953, -3.8800697326660156, -0.8881410956382751, -1.5926315784454346, -1.9399195909500122, -2.023081064224243, -1.9328123331069946, -0.8268134593963623, 0.9665127396583557, -2.2899091243743896, 0.4271523952484131, 0.048356812447309494, -0.9992132186889648, 1.2465243339538574, -1.0505291223526, 4.284538745880127, -2.3981151580810547, -1.2131201028823853, 0.3808063864707947, -0.039169229567050934, 2.408830404281616, 1.2373300790786743, 2.1173722743988037, 0.6751832365989685, 4.195244789123535, -1.3047555685043335, -2.536454200744629, -1.0059654712677002, -1.2456080913543701, 1.6227233409881592, -1.5708959102630615, 2.5129735469818115, 1.4782652854919434, 0.305193692445755, -0.7691487669944763, 1.3050553798675537, -2.362795352935791, 0.5726147890090942], "sim": 0.7659967541694641, "freq": 112485.0, "x_tsne": 55.495819091796875, "y_tsne": -160.03558349609375}, {"lex": "vaccination", "type": "nb", "subreddit": "conspiracy", "vec": [-2.142847776412964, 0.4656095504760742, 1.3258556127548218, 0.6172814965248108, 1.4427776336669922, -0.24071821570396423, -1.6638011932373047, -0.37755006551742554, 0.5841280817985535, 0.7475307583808899, 2.0563406944274902, 1.3986883163452148, 3.4651029109954834, -1.5636600255966187, 2.9770114421844482, 3.6238601207733154, 0.20456776022911072, -1.69566810131073, 0.4933919906616211, 1.569506287574768, -0.46530982851982117, 1.1380836963653564, -1.3845645189285278, 0.4152263104915619, -0.15513209998607635, 2.264134645462036, 1.6042650938034058, -0.11642462015151978, -0.6813538074493408, 1.6984689235687256, 0.9033122062683105, -3.0837864875793457, 0.7519251704216003, -2.9578733444213867, 0.09758812189102173, 0.6719212532043457, -1.5845049619674683, -0.8641021251678467, -2.6329429149627686, 0.6189503073692322, -0.669930636882782, 1.8500699996948242, -1.4748376607894897, 0.7287728786468506, 1.147212266921997, -0.06835253536701202, 0.8317958116531372, 1.2557915449142456, 2.2205097675323486, -1.7937167882919312, -0.2404189258813858, -2.3921267986297607, -1.9832653999328613, -0.04686379432678223, -1.4544825553894043, 0.037368543446063995, 0.5234358310699463, -0.5546719431877136, 1.3742339611053467, 0.7142839431762695, -1.855176568031311, 2.622101068496704, -1.7959660291671753, 1.5031079053878784, 1.054208517074585, 0.41619807481765747, -2.981374502182007, 1.1272509098052979, -2.4142045974731445, 1.542194128036499, -1.5812656879425049, -1.9955393075942993, 0.6279122829437256, 1.5408496856689453, -1.9642131328582764, -3.288442850112915, -0.9802380800247192, -1.3517550230026245, 0.023692350834608078, -1.6224768161773682, 0.9845655560493469, 1.1042523384094238, 0.5438859462738037, -0.22638565301895142, -2.9680073261260986, 2.2654764652252197, -1.5769896507263184, -1.97011137008667, 0.6150739789009094, 0.6918749809265137, 1.8738375902175903, 0.5351380109786987, 2.7955009937286377, 1.488852858543396, 0.40934792160987854, -1.4710909128189087, 0.4975152909755707, 0.8332212567329407, -0.6221139430999756, 0.3415237069129944, 0.5802168250083923, -1.9556032419204712, 0.08460164815187454, -2.3015310764312744, -1.5663912296295166, -2.4553754329681396, 1.3625200986862183, -2.472501516342163, 1.7369277477264404, 1.5974122285842896, 0.45007503032684326, -3.382200241088867, -1.3597469329833984, -0.7941111326217651, -0.5006210803985596, -1.254095435142517, -0.144770085811615, 2.237285614013672, 2.5585451126098633, 0.2231142520904541, 0.15453557670116425, -1.6336783170700073, 1.5034008026123047, 2.54797101020813, -1.1096519231796265, 2.112149715423584, 1.267746925354004, -0.9281826615333557, -0.20687849819660187, 2.284173011779785, -0.4765976071357727, 1.5299665927886963, 0.9453964829444885, -1.5340956449508667, 1.83623468875885, -1.439097285270691, -1.677415132522583, -1.7705330848693848, -0.8198265433311462, 0.20547249913215637, 0.8020827174186707, -1.075795292854309, -1.0462392568588257, 2.347322702407837, 2.8306479454040527, -3.3145980834960938, 0.2589057981967926, -1.856299638748169, -0.39728081226348877, 1.5326581001281738, -0.26238173246383667, 1.4940770864486694, -2.593498468399048, -2.062211513519287, -2.608267068862915, -2.37680983543396, -1.7777812480926514, 0.7001908421516418, -1.5944335460662842, 2.6264750957489014, -0.43545329570770264, 2.22326397895813, -0.08155272156000137, 1.0175197124481201, -1.184827208518982, -3.0210788249969482, 1.2491319179534912, 2.1632678508758545, -0.22049398720264435, 0.3275471031665802, -2.1601078510284424, 0.951082170009613, -0.5772780179977417, -0.24005407094955444, -1.5800282955169678, 1.151733160018921, -0.7340776920318604, -0.3107186555862427, -1.8401544094085693, -0.1372755765914917, -0.6659963130950928, 2.4907169342041016, -0.8650892972946167, -3.5797812938690186, -0.7000254988670349, -1.965806245803833, 0.940325915813446, -0.5756651163101196, -1.756077766418457, -0.6041930913925171, -1.6139469146728516, 1.4070003032684326, 0.01919367164373398, 0.4636716842651367, 1.3349086046218872, -0.97459477186203, -2.389249801635742, -0.7120972275733948, 0.3487018644809723, 2.249614715576172, -1.7095224857330322, 0.27789390087127686, -2.4095730781555176, 1.6924055814743042, -2.600385904312134, 0.5573881268501282, -0.7763316631317139, -1.2082691192626953, 0.3834369480609894, -0.6563749313354492, 2.1486475467681885, 0.5376868844032288, -0.9418982863426208, 1.6182888746261597, 2.6902761459350586, 1.1937228441238403, 0.12384578585624695, -1.1189502477645874, 0.7243541479110718, -1.79877507686615, -1.56222665309906, 0.03343072533607483, 0.10767047852277756, 0.9359241127967834, -3.480252742767334, -0.8055601716041565, -2.4709079265594482, 1.8716121912002563, 0.6833196878433228, 1.4114714860916138, 1.3177375793457031, -0.8439340591430664, -2.0007951259613037, 0.46147972345352173, 2.453557014465332, 0.4383239150047302, -0.23175399005413055, 2.545940399169922, -3.5335443019866943, 0.590643048286438, 0.9327716827392578, -0.11318804323673248, -1.0815309286117554, -0.4804856777191162, 0.39108747243881226, 2.6809678077697754, 2.287217855453491, -1.369652271270752, -0.3369285464286804, 0.29365646839141846, 1.3367177248001099, -1.134617805480957, 0.9605386853218079, 2.9040720462799072, 0.6490116715431213, -0.4154084622859955, 0.5811812281608582, 0.4200470745563507, -0.5994212627410889, 1.125882863998413, -0.4486773610115051, 0.6456625461578369, -1.5261797904968262, -2.295896530151367, -0.6398518681526184, -1.1966131925582886, -1.2094656229019165, -1.4013862609863281, -2.1986682415008545, -0.7631802558898926, 0.46248146891593933, -4.660678386688232, -2.067256212234497, -0.6825658082962036, -1.6236979961395264, -1.4989137649536133, -0.6370455622673035, 2.463306427001953, -2.445557117462158, -0.5683359503746033, -0.5460976362228394, 0.5795335173606873, 0.5845956802368164, 3.25882625579834, 1.469075322151184, -1.1808812618255615, 2.331716299057007, -1.8352349996566772, -2.4487123489379883, 1.516947865486145, 1.0820696353912354, 2.4211692810058594, 1.838605523109436, 1.9431240558624268, 0.1376454085111618, 0.5269595384597778, -0.43651777505874634, -1.2414766550064087, -2.2006311416625977, -0.3796440660953522], "sim": 0.6409702897071838, "freq": 7780.0, "x_tsne": 9.717317581176758, "y_tsne": -143.6610565185547}, {"lex": "treatments", "type": "nb", "subreddit": "conspiracy", "vec": [1.021315097808838, -1.361846923828125, -1.400536060333252, -2.434459924697876, -0.5703791975975037, 1.4295917749404907, -1.2991799116134644, -0.2435755431652069, -0.13974817097187042, 0.7332035303115845, 1.0921123027801514, 0.851379930973053, 3.3993239402770996, -1.2108564376831055, 2.0221457481384277, 2.053049087524414, 0.05141559988260269, 0.04099060222506523, 2.883563756942749, 2.2474637031555176, -1.391065001487732, 2.7015066146850586, -0.4836387634277344, -0.8144513368606567, 3.2101027965545654, 1.5347862243652344, 0.4007505774497986, 0.33452680706977844, 0.6195148229598999, -0.6575822234153748, 0.43066543340682983, 1.0771498680114746, 1.3711599111557007, -2.266524076461792, 0.8973456025123596, 0.05129183828830719, -2.409979820251465, -0.6780288219451904, -0.7154347896575928, -0.22942045331001282, -1.2079057693481445, 1.4752639532089233, -1.1567472219467163, 2.1844398975372314, 1.2380261421203613, 0.6789430379867554, 2.444443941116333, 1.478320598602295, -0.1795746237039566, -1.869710922241211, -0.25827106833457947, -0.11776429414749146, -0.8633548617362976, -0.7575094699859619, -2.4506895542144775, -0.19472786784172058, 0.8590903282165527, -0.489801824092865, -0.38128724694252014, -1.3847020864486694, -0.3725960850715637, 0.09177503734827042, -0.8607593178749084, 3.2514755725860596, 1.668392300605774, 0.8265348076820374, -0.5138211846351624, 1.146991491317749, -0.17189455032348633, 0.29322025179862976, 0.3133603036403656, 0.17036619782447815, 1.456117033958435, -0.6538233757019043, -2.0270230770111084, -1.8361356258392334, -0.4860192537307739, 0.8593481183052063, -0.3784734308719635, 1.5657919645309448, 0.5489771366119385, 1.146280288696289, -1.310581088066101, -0.8269807696342468, -1.3301953077316284, -0.9099901914596558, -2.199984550476074, -0.4052909314632416, -1.6335911750793457, 0.5324259996414185, -0.7268421053886414, -1.5267212390899658, 2.0897579193115234, 1.396252989768982, -2.595529317855835, -2.0943422317504883, 0.3114160895347595, -0.9926924705505371, 1.0627511739730835, -0.2737722396850586, 1.2900967597961426, 1.3663454055786133, 0.9962165355682373, -0.8377400636672974, -3.4982619285583496, -2.2777035236358643, 2.6140570640563965, -1.842005729675293, 1.1081414222717285, 0.3217291235923767, 1.201667070388794, -2.2883129119873047, -1.7283401489257812, -1.9029061794281006, -2.2292745113372803, 0.5305593013763428, 6.345659494400024e-05, 1.1650800704956055, 0.1937456876039505, 0.8338364958763123, -1.0525307655334473, -2.382706642150879, -0.09668129682540894, -1.0088777542114258, 2.291844606399536, 3.7040281295776367, 0.7564034461975098, -2.0928471088409424, 1.359591007232666, -1.767974615097046, -1.726103663444519, -2.860541820526123, 0.7156083583831787, 0.6627148985862732, 2.1940548419952393, 0.3737274408340454, 0.9332743883132935, -1.42667555809021, -0.057638019323349, 1.1515339612960815, 1.125412106513977, -3.292207717895508, -0.01783578097820282, 0.5954757928848267, -0.8996501564979553, -1.9620896577835083, 1.6536059379577637, 0.5005332827568054, 1.5760239362716675, -2.868098497390747, 1.378066062927246, 0.11771225929260254, 3.495518922805786, 0.11213794350624084, 0.8352473974227905, -2.956380605697632, -2.063588857650757, -0.06374649703502655, -1.7751965522766113, -0.29331547021865845, -0.01272410899400711, 1.2878742218017578, -1.3275320529937744, 0.48173463344573975, 1.0844563245773315, -0.2603115439414978, 0.38421592116355896, 1.913582444190979, 0.006310068070888519, 1.0555346012115479, -0.670909583568573, -1.1179789304733276, 2.193110942840576, 1.4738186597824097, 0.4010642468929291, -1.904560923576355, -0.5038260817527771, 1.2858104705810547, -1.0376935005187988, 0.44711631536483765, 1.2225342988967896, 2.761500120162964, -0.702530026435852, 1.101077914237976, -2.5479159355163574, 1.915337324142456, 2.991835117340088, -1.6451252698898315, 1.9087438583374023, -0.5948911309242249, -0.2875775992870331, -1.4421284198760986, 0.75616055727005, 0.7810717225074768, 0.029477659612894058, -1.1653966903686523, -0.7620647549629211, 0.3518642783164978, -0.9808886647224426, 2.7707109451293945, -2.910454034805298, -0.40299755334854126, -1.3116596937179565, 3.4880640506744385, -2.4177157878875732, 1.0066128969192505, -2.1682920455932617, 0.2839745283126831, 2.252288818359375, 0.894087553024292, 0.6683259010314941, -2.2370517253875732, -1.8589730262756348, -0.250837504863739, -1.6898270845413208, -0.5641496181488037, -0.93385910987854, -1.325648546218872, 0.24745887517929077, -1.7966786623001099, -0.14913885295391083, 1.836408257484436, -0.6484671235084534, -0.5711158514022827, 0.09386884421110153, -2.8756890296936035, 2.1769542694091797, -1.4135918617248535, 1.0331714153289795, 0.32405880093574524, 0.383794903755188, -1.0590909719467163, 1.4355477094650269, -0.2682499289512634, 1.9123342037200928, 1.535696268081665, 1.3587570190429688, 4.910272121429443, -0.44336116313934326, -0.36480650305747986, -1.9571141004562378, -0.76837557554245, -1.9977437257766724, -0.40967342257499695, -0.7292134761810303, -2.5958967208862305, 2.5013375282287598, 1.361457347869873, 1.5311193466186523, 0.4451943039894104, -0.5728693008422852, -2.3721323013305664, 0.7822403907775879, 1.4364619255065918, 1.748451828956604, -0.23074758052825928, -2.117500066757202, -0.24302834272384644, 0.17148135602474213, -1.8314160108566284, -0.8995170593261719, 1.1768059730529785, -3.2689242362976074, -2.9206836223602295, 0.7076354026794434, -2.032104015350342, 1.0152716636657715, -1.1135765314102173, 1.2662239074707031, 1.574719786643982, 1.9296733140945435, -1.2816630601882935, 0.13599850237369537, 0.1436782330274582, 3.5378830432891846, 0.18922561407089233, -0.254067599773407, 2.5916543006896973, -0.4578661322593689, -0.9113171696662903, -0.580738365650177, -0.9585356116294861, -1.491817593574524, 1.3527894020080566, 2.3301267623901367, -0.31130629777908325, 1.7339154481887817, -2.2552497386932373, -1.5695719718933105, 1.2371482849121094, 2.3434958457946777, -1.9950448274612427, 1.8300548791885376, 1.7146987915039062, 1.694972276687622, -1.210245966911316, -2.4633991718292236, -0.27306053042411804, -0.40026021003723145, -1.6843945980072021], "sim": 0.6152223348617554, "freq": 4874.0, "x_tsne": 137.416015625, "y_tsne": -36.67171859741211}, {"lex": "medications", "type": "nb", "subreddit": "conspiracy", "vec": [1.028672695159912, -1.3632984161376953, 1.4662432670593262, -0.9919406771659851, -2.4256463050842285, 1.6664433479309082, 1.1193835735321045, 0.7884502410888672, 1.4750522375106812, 0.2853986620903015, 1.2000081539154053, -0.3042258620262146, 2.663545608520508, -1.4128329753875732, 2.4141807556152344, -0.846744954586029, 0.22165775299072266, 0.30600547790527344, 1.3781492710113525, 0.6001051068305969, -1.9536055326461792, 2.1995205879211426, -0.6862892508506775, -1.3345277309417725, 3.7688663005828857, 0.24729014933109283, 0.46748775243759155, 2.3691558837890625, -0.8052233457565308, -0.11188283562660217, 0.9215682744979858, 0.5740393400192261, 1.0985442399978638, -2.3671419620513916, 1.762874722480774, 0.6281395554542542, -0.8657655715942383, -0.08680932223796844, -0.09021451324224472, -2.215561866760254, -0.8838510513305664, 0.32774609327316284, 0.3637806177139282, 3.6161253452301025, 0.4232705235481262, 1.3752671480178833, 3.0282275676727295, 0.008664533495903015, 0.03947651386260986, -0.6588435173034668, -0.40296122431755066, 1.3008232116699219, -0.2922467291355133, 0.6034280061721802, -3.279188632965088, -0.0628543570637703, 0.14641398191452026, -0.24564595520496368, 0.4102816879749298, -1.3126999139785767, -0.5449385046958923, 0.9716424942016602, 0.1316622793674469, 2.1615617275238037, 2.406458616256714, -0.10505939275026321, -0.08702506124973297, 0.42708656191825867, -0.6336888074874878, 1.8549271821975708, 1.3829067945480347, 0.6531718969345093, 0.8285718560218811, -0.8135932087898254, -2.17737078666687, -3.0793771743774414, -1.0000258684158325, -0.35576239228248596, -0.2895241677761078, -0.842008650302887, -0.14041543006896973, 1.2657971382141113, -0.623590886592865, 0.7569673657417297, -1.5570929050445557, -0.10472625494003296, -2.482078790664673, 0.10661346465349197, -1.37924325466156, 1.4865211248397827, -0.7205095887184143, -2.783895254135132, 2.097127676010132, 2.6954472064971924, -4.362605571746826, -0.918481171131134, 1.4947279691696167, -0.7360388040542603, 0.45640116930007935, 0.7731846570968628, 1.1968848705291748, -0.19163711369037628, 0.04077976942062378, -0.5516083240509033, -3.2830991744995117, -2.993906021118164, 0.6592553853988647, -1.703253984451294, -0.02409614995121956, 0.6685804724693298, 1.4185553789138794, -2.896421194076538, -0.785329282283783, -1.960854411125183, -2.7678143978118896, -0.3926065266132355, 0.2809174954891205, -0.8726903796195984, 0.5610573291778564, 1.0743831396102905, -0.44517725706100464, -1.8235702514648438, 0.16146227717399597, -1.2998316287994385, 2.729285717010498, 2.489283800125122, 0.14715683460235596, -1.6701843738555908, 0.8302029371261597, -0.3779723048210144, -0.8963526487350464, -0.28921273350715637, -1.0814367532730103, 1.086775302886963, 1.2630194425582886, -1.0322000980377197, 1.0468120574951172, 0.3835928440093994, 1.855607032775879, 0.8939489126205444, 1.1013948917388916, -2.0618128776550293, -1.1872942447662354, 0.04894368350505829, -1.3328043222427368, -2.0680224895477295, -0.5360285043716431, -0.2516554296016693, 1.0827255249023438, -2.1810195446014404, 0.9551389813423157, -0.10345953702926636, 2.8287971019744873, 0.18869434297084808, 0.8006073236465454, -1.7447761297225952, -0.3801441192626953, 0.16343453526496887, -1.6380813121795654, 0.19348381459712982, 0.7503076791763306, 0.20565655827522278, -1.486052393913269, 1.533263087272644, -0.183519646525383, 1.199265480041504, -0.659866452217102, 2.0462019443511963, -0.26744598150253296, 1.941401481628418, 0.2336047887802124, -0.2537996768951416, 1.026490330696106, 2.1299543380737305, 1.1774711608886719, -2.042789936065674, -1.6774383783340454, 0.9416605234146118, -0.18561074137687683, 0.2909952998161316, -0.16665787994861603, 1.676439642906189, 0.012788631021976471, -0.7120250463485718, -0.5630629658699036, 0.8369656205177307, 1.1448789834976196, -0.9020776152610779, 0.9544100761413574, -0.05637763813138008, -0.14059831202030182, -0.7753537893295288, 1.685854434967041, 0.6379706859588623, -0.6330018639564514, -0.7183161973953247, 0.1627582311630249, 1.0203680992126465, 1.138776421546936, 0.5394043922424316, -1.6347724199295044, 0.4818789064884186, 0.8057177662849426, 4.227853298187256, -2.9189887046813965, -0.9844878911972046, -0.5726436972618103, 0.06665708869695663, 2.2836756706237793, 0.3893965184688568, -0.544424831867218, -2.3202381134033203, -3.507891893386841, -0.6131935119628906, -0.72083979845047, -1.66673743724823, -0.8807455897331238, -1.4766901731491089, -1.470804214477539, -1.2678176164627075, -0.39644452929496765, 0.4401358366012573, 0.6008431911468506, -0.19748815894126892, -0.4810764193534851, -2.5643937587738037, 1.395092248916626, -0.49735796451568604, 0.10323100537061691, 0.22306346893310547, 1.5183310508728027, -0.4505515396595001, -0.721085250377655, 0.32759636640548706, 3.107044219970703, -0.7932322025299072, 0.3112364709377289, 2.0240859985351562, 1.2149302959442139, -0.9995247721672058, -1.0870614051818848, -0.2445186823606491, -2.652914047241211, -2.3919482231140137, 0.22112756967544556, -2.103276014328003, 1.8178889751434326, 0.4898908734321594, 0.896172821521759, 0.4629549980163574, 1.3647170066833496, -0.8599427938461304, 1.3327951431274414, 0.722044825553894, 2.036480665206909, 1.1106208562850952, -1.3581244945526123, -0.011999618262052536, 0.5964678525924683, -1.4684638977050781, -0.2662014067173004, -0.35385701060295105, -2.433997392654419, -3.1542515754699707, -0.21559157967567444, -2.52175235748291, -0.6872955560684204, 0.13374614715576172, 0.3414565920829773, -0.1939711719751358, 0.2206539511680603, 0.09714657068252563, 0.5034352540969849, 0.64895099401474, 1.660341501235962, -0.754276692867279, -1.6213022470474243, 2.775897264480591, -1.801367163658142, 1.1859618425369263, -1.1266984939575195, -1.056291103363037, -2.6876306533813477, 1.6439250707626343, 0.4299487769603729, 0.13098327815532684, 0.8983264565467834, -1.2224135398864746, -1.9671190977096558, -0.4653881788253784, 0.4779130816459656, -0.0942792072892189, 1.6858112812042236, 1.1230480670928955, 2.219806671142578, -0.27618542313575745, -0.2251386046409607, -0.615115225315094, -0.5021103620529175, -0.39528709650039673], "sim": 0.5976815819740295, "freq": 1614.0, "x_tsne": 130.4356689453125, "y_tsne": -55.3960075378418}, {"lex": "vx", "type": "nb", "subreddit": "conspiracy", "vec": [0.17377617955207825, 0.4293699860572815, 0.5432334542274475, 0.17237013578414917, 0.040634166449308395, -0.764417827129364, -0.3875097334384918, -0.03928658366203308, 1.183302640914917, 1.429417371749878, 0.5830217599868774, -0.3283330202102661, 1.421429991722107, -0.6683833599090576, 1.3366209268569946, 0.7984417080879211, 0.1855766475200653, -0.09014290571212769, 0.3548837900161743, 1.4608697891235352, 0.5069380402565002, 0.035622868686914444, -0.1713942289352417, -0.15948347747325897, 0.6943579316139221, -0.03352607786655426, 0.17785684764385223, 0.2616439461708069, 0.16516633331775665, 0.478288471698761, 0.7375435829162598, -0.02865171805024147, 0.660408079624176, -1.5236783027648926, 0.534697949886322, 0.5115615725517273, -0.6326978206634521, 0.12883427739143372, -1.2650386095046997, 0.7312098741531372, -0.39097070693969727, 1.1189604997634888, -0.6224710941314697, 0.29312172532081604, 0.5164632797241211, -0.7774680256843567, 1.3425339460372925, 0.338493287563324, 0.5950399041175842, -0.2981013059616089, -0.10316895693540573, -2.2314956188201904, -0.2050076276063919, -0.8302913308143616, -0.8735044598579407, -0.43086111545562744, 0.05179319158196449, 0.561247706413269, -1.0249744653701782, -1.7311527729034424, 0.1607843041419983, 1.4355823993682861, -0.4149010479450226, 1.0475225448608398, -0.2452816516160965, -0.3631760776042938, -0.7711788415908813, 0.18821759521961212, -0.18330539762973785, -0.4370589554309845, 1.3566279411315918, -0.2550782561302185, -0.27327197790145874, 0.20353616774082184, -1.576972484588623, -1.8759957551956177, 0.22426271438598633, 0.15689411759376526, -0.9947802424430847, -0.08361060917377472, 0.32341504096984863, 0.4395388066768646, -0.4842081665992737, 1.0302886962890625, -0.9924162030220032, 0.35003000497817993, -0.7670870423316956, -0.5987118482589722, -0.6241986155509949, 0.6180034279823303, 0.45114994049072266, 0.7468964457511902, 1.5745575428009033, -0.18314148485660553, -0.6284576058387756, -0.009250383824110031, 0.24534080922603607, -0.19002997875213623, 0.2910146117210388, 0.45093655586242676, -0.21454569697380066, -0.3921052813529968, 0.1907251626253128, -0.37700891494750977, -0.6347097158432007, -1.0251696109771729, 1.3655062913894653, -0.4978411793708801, -0.7103367447853088, 0.3628600537776947, -0.05678914487361908, -1.1374342441558838, -1.2918978929519653, -0.3596407175064087, -0.6121813058853149, 0.114918053150177, 0.080381840467453, -0.3394727408885956, 0.3434901237487793, -0.07676728814840317, 0.5431684255599976, -0.153607577085495, -0.4511749744415283, 0.24370770156383514, -0.4114310145378113, 1.1823506355285645, 0.21927277743816376, -0.7344375848770142, 0.21607080101966858, 0.45795366168022156, -0.2509825825691223, 0.08805511146783829, 0.45798999071121216, 0.3425747752189636, 0.4160764515399933, -1.2523748874664307, -0.3809545636177063, -0.7760767936706543, 0.4631321430206299, 0.7702264785766602, -0.8378543853759766, -0.8291700482368469, -0.4351712167263031, 1.3062021732330322, 1.3994734287261963, -1.1044983863830566, -0.33193346858024597, -0.33041948080062866, 0.417182058095932, -1.042432427406311, 0.6051616072654724, 0.5176810026168823, -1.7987866401672363, 0.1297880858182907, -0.13956335186958313, 0.611624002456665, -0.9137808680534363, 0.22347351908683777, -0.5025383830070496, 1.486405372619629, -0.4205063283443451, 0.9646900296211243, -0.09472409635782242, 0.5661656260490417, 0.16767020523548126, 0.04594627022743225, 0.4698396921157837, 0.7825394868850708, -0.3765338659286499, 1.0751458406448364, -0.10880860686302185, -1.6274913549423218, -1.1206977367401123, 0.06176493689417839, -0.5375940799713135, -0.0742495208978653, -0.3288143277168274, -0.5644912719726562, -0.5963723063468933, 0.19297538697719574, -0.55877685546875, 0.26910391449928284, 1.0223026275634766, 0.013777323067188263, 0.8456808924674988, 0.3438844084739685, 0.9965356588363647, 0.22908681631088257, 0.6979756355285645, 0.4459058940410614, -0.4896426200866699, -0.032597824931144714, -0.34759700298309326, 0.1452152281999588, -0.2833356261253357, -0.31692367792129517, -0.7974305748939514, -0.2133864313364029, -0.1541348546743393, -0.3293609321117401, -0.42248353362083435, 1.2981529235839844, -0.03207232803106308, 0.9143809676170349, -1.402724027633667, -0.2647871971130371, -0.42081236839294434, -0.4979912340641022, 0.46698641777038574, -0.5680783987045288, -0.2776715159416199, 0.23491787910461426, -0.8335630893707275, 0.4777843952178955, -0.0886698067188263, 0.2245127260684967, -0.2845644950866699, -0.11973795294761658, -0.8239485621452332, -1.4975485801696777, 1.1239821910858154, 0.462817907333374, -0.29016128182411194, 0.0004991143941879272, -0.7078741192817688, -0.5239260196685791, 0.009912708774209023, 0.8224684596061707, 0.3943849503993988, 0.6075418591499329, 0.022067628800868988, -0.8016481995582581, -0.2139650136232376, 0.48710474371910095, 1.2873129844665527, -0.5636292695999146, -0.7964632511138916, 1.370882272720337, 0.2868598997592926, 0.7171898484230042, 0.23205529153347015, -0.2873600423336029, -0.43396472930908203, -0.862739086151123, 0.054031819105148315, 0.013128474354743958, 0.5591838359832764, 0.09693725407123566, 0.6751003265380859, 1.193483829498291, -0.17028063535690308, -1.6181124448776245, 0.995379626750946, 0.6826284527778625, 1.3065736293792725, 0.09828086197376251, -1.130519986152649, -0.1657407283782959, 0.8502075672149658, 0.04399450495839119, 0.6857627034187317, -0.524416446685791, -0.9281033277511597, -1.3003740310668945, 0.3537541925907135, -0.21345533430576324, -0.06339170038700104, 0.5930736064910889, -0.5965123772621155, 0.7010467052459717, 0.9938004016876221, 1.0232064723968506, -0.5799399018287659, -0.7285897731781006, 0.07201569527387619, 1.4636019468307495, 0.20724862813949585, 2.6083033084869385, 0.12965410947799683, -0.6057460904121399, -0.47695061564445496, -1.4920928478240967, 0.5556256175041199, 0.9665848612785339, -0.14099329710006714, -0.040963493287563324, 1.4260737895965576, -1.6838992834091187, 0.0036095078103244305, -0.30792468786239624, 0.007850755006074905, 0.4892580807209015, -0.6407209038734436, 0.7713220715522766, 0.9479565024375916, 0.5643211603164673, -0.8493070602416992, 0.3418762683868408, -0.3981941044330597, -0.17149001359939575], "sim": 0.5895665287971497, "freq": 68.0, "x_tsne": -19.13440704345703, "y_tsne": -55.22367477416992}, {"lex": "vax", "type": "nb", "subreddit": "conspiracy", "vec": [-1.4118965864181519, 0.773285448551178, 1.4240074157714844, -0.024890411645174026, -0.7439469695091248, -1.1316252946853638, 0.024942750111222267, 1.033849835395813, 1.3265713453292847, 1.496480941772461, 0.5815696120262146, 1.7646843194961548, 2.943042755126953, -0.06076373904943466, 2.9385299682617188, 1.078209638595581, -0.0331290140748024, 0.5788146257400513, 0.0027191247791051865, 3.3815200328826904, 1.304900884628296, -0.041658103466033936, -0.7641941905021667, 3.1139280796051025, 1.9953874349594116, 0.18920041620731354, 1.1431005001068115, 0.15047422051429749, 0.9617750644683838, -0.4539986252784729, -0.4688766598701477, -2.0230584144592285, 0.6420740485191345, -1.847301959991455, -1.7123539447784424, 0.1861087530851364, -0.3877590298652649, -1.3046565055847168, -1.5018988847732544, 1.5632740259170532, -0.31768614053726196, 1.7019611597061157, -1.1108688116073608, 0.8571054935455322, -0.3344786763191223, -0.4929834008216858, 0.9638220071792603, -1.0400502681732178, 1.8039116859436035, -0.7390514612197876, 0.058006346225738525, -1.751758098602295, -0.24951419234275818, -1.1390209197998047, -2.256960391998291, -1.4469789266586304, 1.4883660078048706, -1.1366167068481445, -0.05391053482890129, -2.6311652660369873, -1.7917195558547974, 0.4956132471561432, -2.4958789348602295, 0.5513001680374146, 2.413804531097412, 0.2536747455596924, -1.5158590078353882, 0.5815445780754089, -0.6350641846656799, 2.8011386394500732, 0.9871029853820801, 1.2486588954925537, 3.153686046600342, 2.5266449451446533, 0.2627463638782501, -2.8598155975341797, -0.09564226865768433, -0.3006149232387543, 0.9510571956634521, -0.40648186206817627, -0.7309732437133789, -1.2590537071228027, 0.0891551524400711, 0.8350855708122253, -1.140699028968811, 0.8923125863075256, -2.8824048042297363, -2.203340768814087, -1.5217585563659668, 1.2444483041763306, -0.7327026724815369, 0.6265493631362915, 1.6255635023117065, 1.694400668144226, -0.03775341436266899, -0.11692193895578384, -0.3427084684371948, -0.34814780950546265, 0.24550411105155945, -0.3726177215576172, 1.1084840297698975, -2.772240161895752, 0.09483025223016739, -1.2111988067626953, -1.3203693628311157, -4.36393404006958, 0.8341937065124512, -1.6080862283706665, 2.5330376625061035, 0.8571996688842773, 0.938429594039917, -4.250924110412598, -1.0253171920776367, 0.5794392228126526, -1.0876758098602295, -0.5737947225570679, 0.6576617956161499, 0.8439871072769165, 2.0682291984558105, -1.1455360651016235, 0.740355908870697, -0.8327929973602295, 1.3950384855270386, 0.39251503348350525, 1.2197051048278809, 2.210547685623169, -0.08855326473712921, -2.5968503952026367, -0.2781519591808319, 1.4701051712036133, 0.6713821291923523, 0.200083389878273, -0.4220018684864044, -1.139009952545166, -0.07146672904491425, -0.2943950295448303, -2.818870782852173, 0.5795760154724121, 0.3306010961532593, 0.6469826102256775, 1.702647089958191, 0.008165779523551464, -1.2072038650512695, 1.7584781646728516, 0.15905234217643738, -3.776787281036377, 1.4280328750610352, -0.5263541340827942, 1.2893034219741821, 0.3974636495113373, -1.0296204090118408, 0.35556769371032715, -1.8119984865188599, -2.038665294647217, -0.3879850506782532, -0.4688027799129486, -0.11791551113128662, 1.395327091217041, -2.5880563259124756, 1.8729500770568848, 0.131637305021286, -0.21881216764450073, -1.8690012693405151, -0.3926088809967041, -1.504699468612671, -1.9558765888214111, 0.9833474159240723, 1.7029470205307007, -0.7948495149612427, -0.5779286623001099, 0.23770099878311157, 0.11636201292276382, -0.6276642084121704, -1.6597483158111572, -0.4972302317619324, -0.04032871127128601, -1.5753751993179321, -1.0520051717758179, -1.7648018598556519, 1.4367527961730957, -2.0947959423065186, 1.3040651082992554, -1.6199630498886108, -0.395221084356308, 2.1687333583831787, -2.7542498111724854, 0.8192912936210632, 0.16789136826992035, 0.07661855965852737, -0.13703452050685883, -1.8074769973754883, 0.2303999811410904, 0.23933596909046173, 0.1856025755405426, 0.5019038319587708, 0.1959516704082489, -3.11319899559021, -1.3326964378356934, -0.6963948011398315, -0.32101675868034363, -1.8727219104766846, 1.1314882040023804, -0.4275397062301636, -0.331965833902359, -1.4841614961624146, 1.1379636526107788, -1.9164823293685913, -2.7815446853637695, 1.045719861984253, -1.2166883945465088, 1.0387719869613647, -0.08607262372970581, -0.4256570339202881, -0.5943644046783447, 1.5377615690231323, 0.05472393333911896, -0.6222245693206787, -0.23506522178649902, 0.34616559743881226, -0.6112428307533264, -1.2095757722854614, 1.115503191947937, 0.8362638354301453, 0.48152220249176025, -0.7396036982536316, -1.3375415802001953, -1.1954257488250732, 2.236168622970581, -0.43626996874809265, 0.5318244099617004, 1.3299182653427124, -1.6845118999481201, -0.21509504318237305, 1.5897746086120605, 2.637982130050659, -0.2538682520389557, 0.1863158494234085, -0.08950284123420715, -0.44749340415000916, -0.6269341111183167, 1.0810412168502808, -0.10790223628282547, 1.56149160861969, -1.3177893161773682, 1.258518934249878, 1.2200976610183716, 0.47500482201576233, 0.24639827013015747, -0.9227054119110107, 1.4703582525253296, 0.5657333731651306, -0.07797392457723618, -0.7559040784835815, 0.9899237751960754, 0.6653634309768677, 0.5598264932632446, -1.0904366970062256, -0.11804867535829544, -3.0254604816436768, 1.1558160781860352, -0.8689565062522888, 0.5791965126991272, -1.4061236381530762, -1.7872148752212524, -1.162611961364746, -0.5855363011360168, -1.5731467008590698, -0.8762386441230774, -0.14126139879226685, -0.9579590559005737, -0.9253882169723511, -2.9914212226867676, -2.2601590156555176, 0.24146811664104462, -0.019816569983959198, -1.3086944818496704, -1.5501986742019653, 1.771836519241333, -3.66208815574646, -1.1130515336990356, 1.71529221534729, 0.31778621673583984, 1.8033714294433594, 1.450615406036377, 0.15858377516269684, -0.009760705754160881, 2.570234537124634, 1.325573205947876, -1.9233341217041016, 1.2239714860916138, 0.052179597318172455, 0.6715133190155029, -0.3171519339084625, 0.7305852174758911, 0.7250350713729858, 1.8794288635253906, 1.0554535388946533, 1.8235208988189697, 0.6740672588348389, 2.4785239696502686], "sim": 0.5812548995018005, "freq": 3208.0, "x_tsne": -84.93602752685547, "y_tsne": -139.11485290527344}, {"lex": "injections", "type": "nb", "subreddit": "conspiracy", "vec": [0.06099848821759224, -0.8551937341690063, 0.3800974190235138, -1.629931092262268, -1.1483738422393799, 1.1974198818206787, -0.20243792235851288, 0.7565815448760986, 1.2732692956924438, 1.9715908765792847, -0.0212043859064579, 0.6719478964805603, 3.2587850093841553, -1.3465137481689453, 2.031238555908203, 3.5472195148468018, 0.9220677614212036, -1.9451173543930054, 1.6164445877075195, -0.7960370779037476, -2.4379608631134033, -0.25356775522232056, -3.3406758308410645, -0.8562955856323242, 1.8797523975372314, -1.2382689714431763, 0.95954829454422, 2.6534061431884766, -0.9884033799171448, -0.04724165052175522, -0.09552659094333649, 0.024842839688062668, -1.4440865516662598, -2.751260280609131, -0.2564601004123688, 0.9617013931274414, 0.1982840597629547, -0.5861110687255859, -1.6956003904342651, -0.3444380760192871, -1.3064881563186646, 1.558152198791504, 0.2945403754711151, 0.7333990335464478, 0.7942218780517578, -0.2905413806438446, 0.31402045488357544, 1.301173448562622, 2.8408892154693604, -1.6767504215240479, 2.197315216064453, 0.24599429965019226, -0.006224019918590784, -0.264256089925766, -1.88924241065979, -1.245235562324524, 0.5126166939735413, -1.1319193840026855, 0.9220049977302551, -1.4419430494308472, 0.685199499130249, -1.271816372871399, -2.143131971359253, 0.8220434784889221, -0.3484998345375061, 1.9845147132873535, -1.2623498439788818, -1.0364539623260498, 1.1189851760864258, 0.5723890066146851, -0.7066687345504761, 0.16061021387577057, 0.23992007970809937, 0.7028838396072388, -0.4693003296852112, -2.179713726043701, -0.25450119376182556, 0.4740436375141144, 0.6524868607521057, -1.5183792114257812, 1.5272266864776611, -0.5850588083267212, 0.9503586292266846, 1.8406133651733398, -0.27659156918525696, 0.9767366051673889, -1.3738967180252075, -0.6830477118492126, -1.1452295780181885, 0.49221310019493103, -0.6505947709083557, -0.01656436175107956, 1.311461091041565, 0.7521792650222778, -3.6008074283599854, -1.2159441709518433, -0.48938825726509094, 1.304780125617981, -0.9972131252288818, -0.5324285626411438, -0.7410739660263062, -0.5004756450653076, -0.20465752482414246, -1.3335421085357666, -2.599393844604492, -2.0948104858398438, -0.07507532089948654, 0.6819547414779663, 1.0220539569854736, 0.7672092914581299, -1.9381684064865112, -3.1760811805725098, 0.5679175853729248, 0.496060848236084, -2.2941031455993652, 0.33093294501304626, -2.7571001052856445, 0.18430164456367493, -1.268830418586731, 0.4628945589065552, 0.7792508602142334, 1.1983753442764282, 0.0075488220900297165, -0.1112176775932312, 0.6265363097190857, 2.7966456413269043, 1.6987395286560059, -1.4057092666625977, -0.2840428054332733, -1.025670051574707, 0.222787007689476, 0.25607603788375854, -0.06803694367408752, 0.49823468923568726, 1.013062596321106, 0.47666677832603455, 0.12414969503879547, -0.4186444878578186, 0.2028360366821289, -0.1293133795261383, 1.443245530128479, -1.3811746835708618, 0.13237231969833374, -1.074877381324768, -0.42778530716896057, -2.124086380004883, -0.48860329389572144, -1.4922270774841309, 1.0840479135513306, -0.4883454740047455, 0.5596486926078796, -0.7529481053352356, -0.6384070515632629, -0.2435351461172104, -1.2280038595199585, -0.22806651890277863, -1.642943263053894, -0.9045053124427795, -2.439213991165161, 1.2435939311981201, -0.8653636574745178, -0.4444308876991272, -1.1298435926437378, -0.19580279290676117, -1.3979032039642334, -1.64737069606781, -0.8698444962501526, 1.7776049375534058, -1.1917576789855957, 2.056703567504883, 0.21191561222076416, -0.7343695759773254, 0.3147752583026886, -0.5125431418418884, 0.266348272562027, 0.6845390796661377, -0.9339224696159363, -0.7737315893173218, 0.5443516373634338, -0.8636690378189087, -1.1182271242141724, 1.4398528337478638, -1.7323939800262451, -0.5071499347686768, -0.22537216544151306, 0.13814564049243927, 3.331015110015869, -0.7055655717849731, -0.6043514609336853, -0.12113407999277115, -0.9212542772293091, 0.3738241195678711, 1.3208119869232178, 0.47669583559036255, -0.3724712133407593, -2.102010488510132, -0.7748061418533325, 1.2742887735366821, -0.10537523031234741, 0.9316264986991882, -2.371166944503784, 0.6824015974998474, 1.2578957080841064, 2.1763925552368164, -0.5078887343406677, -0.6992433071136475, -1.9768404960632324, -0.048149801790714264, -0.5020573735237122, 0.8723103404045105, 0.18182791769504547, -1.2846394777297974, 0.31374263763427734, -1.031171202659607, 1.590343952178955, 1.106688141822815, 0.22011852264404297, -1.1568880081176758, -0.17534440755844116, 0.5233767032623291, 0.8402553796768188, 1.3099340200424194, 0.8389903903007507, 0.4412640333175659, -0.8456618785858154, -1.3280518054962158, 0.5482400059700012, -0.11827020347118378, 0.8954283595085144, 0.17973215878009796, 1.5731509923934937, -3.2028276920318604, -0.7391659617424011, 1.1933432817459106, 1.5061925649642944, -1.7409871816635132, 1.1963392496109009, 3.150625228881836, -1.3524179458618164, 1.7706314325332642, -1.1396586894989014, 0.2598315179347992, 0.8756234049797058, -1.4038652181625366, -0.5491414070129395, 0.31340664625167847, 1.9413557052612305, 1.239349365234375, -0.11927621066570282, 1.2146985530853271, 1.8180830478668213, -0.44857165217399597, 0.8119980096817017, 1.6034152507781982, 1.74988853931427, -0.30337223410606384, 0.6376908421516418, 0.8423265218734741, -1.3707621097564697, -1.261014461517334, -0.06497526168823242, 2.239922523498535, -0.18570509552955627, -2.1165525913238525, 0.5139747858047485, -0.43750396370887756, -0.5001915693283081, -0.8947345614433289, 0.18239937722682953, 0.41710713505744934, -0.607042133808136, -1.131107211112976, -1.147827386856079, -0.3058265745639801, 2.283283233642578, -1.4933000802993774, -1.9557839632034302, 1.8079370260238647, 0.1725686937570572, -1.1262093782424927, -1.0942224264144897, -1.6591957807540894, 3.2367632389068604, 2.677804946899414, 3.6369218826293945, -0.8242281079292297, 0.3907857835292816, -2.393458366394043, -0.14877834916114807, -0.7834600806236267, 1.0470231771469116, 0.10843860357999802, 0.26420366764068604, 1.9521958827972412, 3.0752506256103516, -0.21327286958694458, -1.0356450080871582, -0.7332110404968262, -1.5817567110061646, -0.3057626485824585], "sim": 0.5693145394325256, "freq": 795.0, "x_tsne": 67.41427612304688, "y_tsne": -47.698734283447266}, {"lex": "adjuvants", "type": "nb", "subreddit": "conspiracy", "vec": [1.2562452554702759, 1.2763181924819946, 0.09926169365644455, -0.8308308720588684, 0.6564953923225403, 1.2675328254699707, 0.8911675214767456, 0.6887330412864685, -0.15416042506694794, -0.3177188038825989, 0.5538603663444519, -0.8046817779541016, 2.959294319152832, -1.7871986627578735, 1.0347152948379517, 0.7417791485786438, -1.0800223350524902, -0.5812963843345642, -0.9364644289016724, 1.53080415725708, -0.7875833511352539, 0.07550514489412308, -1.3444961309432983, -1.029097080230713, 2.1138370037078857, 0.31711092591285706, 0.5329945683479309, 0.6406720280647278, -1.1917893886566162, -0.32608094811439514, -0.8709205389022827, -1.1379345655441284, -0.4552594721317291, 0.3197021186351776, 1.0258007049560547, 0.9938756227493286, 0.27681198716163635, 0.6137345433235168, -2.5556700229644775, -1.546059489250183, -0.7325922846794128, 0.29244521260261536, -0.7220243811607361, 0.37651747465133667, -0.7478268146514893, 0.7316204309463501, 1.9901987314224243, 2.7056593894958496, 0.9132290482521057, -0.06240253522992134, 0.09754463285207748, -1.5489970445632935, -0.27434515953063965, 0.5578536987304688, -0.6281853318214417, -0.20780368149280548, -0.6431285738945007, -1.3062736988067627, 0.8618634939193726, -0.973954975605011, 0.41449758410453796, -0.0849766954779625, 0.19393157958984375, 0.10295096039772034, 0.45300203561782837, 0.7300779819488525, -0.16684342920780182, -0.044861651957035065, 0.6525439620018005, 1.4426863193511963, 2.2603371143341064, -0.41743582487106323, -2.1131081581115723, -0.09296534210443497, -0.017189115285873413, -0.9952301979064941, -0.6009255647659302, 0.2851620316505432, -0.25786811113357544, 2.0113375186920166, 1.4375238418579102, -0.09542527049779892, -2.051138401031494, 1.3091096878051758, -1.9428797960281372, 0.6099293231964111, -0.923220157623291, -1.892007827758789, 0.589917778968811, 1.0787948369979858, -0.38923075795173645, -0.45996832847595215, 4.326098442077637, -0.07597247511148453, 0.15219853818416595, 0.019136015325784683, -0.1609383374452591, 0.9699397087097168, -0.0770554468035698, 0.25432008504867554, 0.11535336077213287, -0.7889058589935303, 0.416679710149765, 0.3145792782306671, -1.585788607597351, -1.33222234249115, -0.8682666420936584, -0.4629787504673004, 0.7775329351425171, 1.2995877265930176, -1.221954107284546, -0.9844711422920227, -1.8256096839904785, 0.8311334252357483, -0.6306309700012207, -0.6704487204551697, -0.7607267498970032, -0.6701487898826599, 0.6697141528129578, 0.5959237813949585, -0.6333765983581543, 0.6721227169036865, -1.3964197635650635, 1.058741807937622, 0.29721179604530334, 4.051595687866211, 0.47627463936805725, -2.1780612468719482, 2.396009683609009, 0.5660766959190369, -1.8041961193084717, 0.1872449368238449, -1.493496060371399, -0.8207793235778809, -0.28296583890914917, -0.43539896607398987, -1.4854309558868408, -2.462031602859497, 0.5841723680496216, -0.024823494255542755, 1.6738355159759521, -0.9954766035079956, 0.035027697682380676, 0.4676668643951416, 0.16885757446289062, -0.465179979801178, -1.2638174295425415, -0.13751846551895142, 1.9155278205871582, -1.2596948146820068, 0.635530412197113, -2.0951931476593018, -0.9685443639755249, 0.18157705664634705, -1.492108702659607, 0.13738203048706055, -1.3991256952285767, 0.4290090501308441, -2.6276652812957764, 0.6127554178237915, 0.8539475202560425, 0.5085460543632507, -0.8808173537254333, 0.4765649437904358, -0.13133661448955536, -1.9245221614837646, -1.1054713726043701, 1.329611897468567, -0.10540522634983063, 2.854531764984131, 0.6514473557472229, 1.5565136671066284, -0.4485936164855957, 0.14787279069423676, 0.7058696746826172, -0.9600042700767517, -1.9999631643295288, 1.5487090349197388, 0.8025280833244324, -0.546489953994751, -0.2848054766654968, 1.0803899765014648, -0.5413633584976196, 1.0484319925308228, -1.867606282234192, 0.28758999705314636, 1.0706408023834229, -0.020564496517181396, 1.2444920539855957, -0.021386701613664627, -0.8647417426109314, -1.0131456851959229, -0.0629388764500618, 0.37795230746269226, 1.605879306793213, -0.34460556507110596, 0.3127632737159729, 0.5850552916526794, 0.9280341863632202, -0.7991582751274109, -0.7455069422721863, -0.08401334285736084, 0.20704761147499084, 0.8605005741119385, -1.2138441801071167, 0.16634082794189453, -2.007377862930298, 0.441045880317688, -0.019928140565752983, 1.4157516956329346, -2.2335891723632812, -1.2787716388702393, -2.1781084537506104, -1.035765528678894, 0.35488465428352356, -0.007273223716765642, -1.4613698720932007, -0.555679202079773, -0.5834753513336182, 1.2850397825241089, 0.6847087740898132, 1.3381948471069336, 1.8348698616027832, -0.7288986444473267, -2.1328237056732178, 0.5341643691062927, -0.6013943552970886, 1.015100121498108, -0.8190394639968872, -0.8793697357177734, 1.2637172937393188, -1.430667757987976, 0.1467730551958084, 1.987852931022644, 2.69226336479187, -0.6269813179969788, 0.23693622648715973, 1.7033909559249878, 0.0211077481508255, -0.5923340916633606, 0.7257028222084045, -0.15142197906970978, -0.6441236734390259, -0.9345884919166565, -1.8474818468093872, -0.30141937732696533, 2.6343915462493896, 1.0987168550491333, 1.5420860052108765, 0.5205232501029968, 0.14705313742160797, -2.655740737915039, 3.756328582763672, 2.036132335662842, 0.6792348623275757, 0.8998965620994568, -0.8972718715667725, -1.1780656576156616, -0.514947772026062, -0.818868100643158, -0.6557908058166504, 0.30820727348327637, -1.3558595180511475, -3.2179622650146484, -0.6533984541893005, -1.6193112134933472, 1.234797477722168, -0.5275998115539551, 0.13296832144260406, -1.0003883838653564, -0.269714891910553, -1.585802674293518, -1.9360847473144531, 0.2994177043437958, 1.0242654085159302, 2.7360713481903076, -0.6117340922355652, 0.9636697769165039, -1.183716893196106, 0.11419318616390228, -1.7476766109466553, 3.0008997917175293, -1.8187885284423828, -0.2735254168510437, 1.2135894298553467, -0.08069657534360886, -0.0010475772432982922, -1.069654107093811, -1.0091972351074219, 0.2623015344142914, 0.25490328669548035, 0.08144599199295044, 1.1248427629470825, -1.072009563446045, 1.3373103141784668, -0.26614537835121155, -0.8822569847106934, 1.2050790786743164, 0.19064415991306305, -0.13065692782402039], "sim": 0.5680330991744995, "freq": 199.0, "x_tsne": 105.84004211425781, "y_tsne": -39.65829086303711}, {"lex": "medicines", "type": "nb", "subreddit": "conspiracy", "vec": [-0.07833635807037354, -0.32620933651924133, 0.50843745470047, -1.8613617420196533, -1.7556443214416504, 2.5002176761627197, 1.0652358531951904, 0.9125286340713501, 0.8045052886009216, 0.33428022265434265, 1.5759966373443604, 0.26150426268577576, 1.6608394384384155, -1.4906102418899536, 3.1116859912872314, -0.7195287942886353, 0.15856899321079254, 1.2363834381103516, 2.682142496109009, -0.21868884563446045, -1.1859136819839478, 0.5527888536453247, -0.6312459111213684, -0.5738369822502136, 4.3348069190979, 0.053116168826818466, 0.783256471157074, 0.7497223019599915, 1.1803230047225952, -1.2318941354751587, 0.30223530530929565, 0.07200165838003159, 0.9402568936347961, -1.233513593673706, 0.7234833836555481, 0.21747425198554993, -2.3844547271728516, 1.0359816551208496, -1.1415830850601196, -0.17298562824726105, -2.361063241958618, 1.3776273727416992, 0.2356162816286087, 2.873380661010742, 1.631900429725647, -0.2996402680873871, 2.0034844875335693, -0.12811070680618286, 0.3527899384498596, -0.2627381682395935, -0.31573745608329773, 0.49493005871772766, 0.611794650554657, 0.11425110697746277, -3.0209100246429443, 0.607046365737915, 0.38395389914512634, -0.8329808115959167, -0.9332091808319092, -1.8276081085205078, -0.3153906762599945, 1.1731488704681396, 0.235122412443161, 1.6240109205245972, 0.2879738211631775, -0.9330037832260132, 0.08239927887916565, 0.6111612319946289, -1.9009854793548584, 0.34834781289100647, 0.24344421923160553, -0.24671904742717743, 0.1388336718082428, -0.41214945912361145, -0.37104034423828125, -3.466709613800049, -0.5005454421043396, 0.007212969474494457, -0.3672725558280945, 1.879823088645935, 1.0552960634231567, 0.18320757150650024, -0.5093724727630615, -0.23623256385326385, -0.699447512626648, -0.9112659096717834, -0.7265675663948059, 0.7137003540992737, 0.00873531773686409, 0.7732279300689697, -0.2680281400680542, -2.0350639820098877, 1.8417917490005493, 2.5111591815948486, -2.760530471801758, -1.402762532234192, 0.5473562479019165, -2.3542728424072266, 0.3670603334903717, -0.08772201836109161, 1.0627330541610718, -0.6959325075149536, 0.389451801776886, -2.187213897705078, -2.416459321975708, -1.323144555091858, 2.5483782291412354, -1.260014533996582, -2.0527966022491455, 0.3090531527996063, -0.5484350919723511, -1.6600645780563354, -1.0058788061141968, -0.2087569385766983, -1.23087477684021, -0.7845180630683899, 1.0136388540267944, -0.38857200741767883, 0.739356279373169, 1.319309949874878, 0.031106792390346527, -2.0445334911346436, -1.0642927885055542, -0.5771098732948303, 2.188466787338257, 2.570622682571411, -0.012662343680858612, -2.252502202987671, 1.9669493436813354, 0.5860474109649658, -1.9970307350158691, -0.2361544370651245, -0.7147200107574463, 0.511760413646698, 1.178196907043457, -0.8619589805603027, -1.0190719366073608, -0.22963570058345795, 2.8111324310302734, 1.8510440587997437, 0.9803084135055542, -2.508228063583374, 1.1132937669754028, -0.36929115653038025, 0.19024480879306793, -0.5407394170761108, -0.02331543155014515, 1.350141167640686, 0.6098411083221436, -3.0841002464294434, 1.7271465063095093, -0.44624465703964233, 3.004754066467285, 0.33885881304740906, -1.1284401416778564, -3.0024876594543457, 0.3364931344985962, -1.1597156524658203, -1.4518444538116455, 0.3205733597278595, 0.8634880781173706, -0.6278083920478821, 0.40979263186454773, -0.2144569307565689, -0.46167999505996704, 2.0464463233947754, -1.727168083190918, 2.188493013381958, -1.9020365476608276, 0.9332548975944519, 0.243976429104805, -0.1541651338338852, 0.7028040885925293, 0.9122844934463501, 0.5573803186416626, -0.6843926906585693, -1.2840687036514282, 1.0825012922286987, -0.930945098400116, -0.22029650211334229, 0.9974617958068848, 2.237417697906494, 3.083724021911621, -0.6636425256729126, -0.8177159428596497, -0.7874522805213928, 2.6998603343963623, -1.2389651536941528, 1.2403271198272705, 1.3518340587615967, 0.037596073001623154, -1.9749196767807007, 0.1656429022550583, 0.14697659015655518, -1.48556649684906, -1.5557342767715454, -1.48931884765625, -0.8075470328330994, -1.0489895343780518, 1.5715585947036743, -2.6914241313934326, -1.1193972826004028, -0.6063201427459717, 2.7230148315429688, -2.613658905029297, -0.5156037211418152, -0.15607640147209167, 0.05706663057208061, 1.3894094228744507, -0.2961167097091675, -1.5194487571716309, -2.0149428844451904, -3.139265775680542, 1.6187069416046143, 0.07199718803167343, -1.6771124601364136, -0.38252612948417664, -2.008234977722168, -1.4195572137832642, -0.8477935791015625, -0.8620020151138306, 1.6229294538497925, 0.05047701299190521, -1.3115822076797485, -0.12589645385742188, -1.570496916770935, 3.436683416366577, -1.4409775733947754, -1.2440571784973145, -1.006447434425354, 1.1785231828689575, 0.7730290293693542, -0.6172260046005249, -0.4173164367675781, 3.148345947265625, -0.7045621275901794, 0.13892008364200592, 2.386676073074341, 0.7842724323272705, -0.7141895294189453, 0.10320714861154556, 0.2476082146167755, -2.430922746658325, -0.9994950890541077, -0.0470258966088295, -1.5680633783340454, 3.7987403869628906, 0.2403159737586975, 2.196758508682251, 0.0854950025677681, 0.26397505402565, -1.4011876583099365, 2.443758249282837, 0.45128703117370605, 1.642698049545288, 0.0622936375439167, -2.2748918533325195, -1.8115200996398926, 0.09878861159086227, -1.4820927381515503, -0.3999786674976349, -1.6276803016662598, -1.9425103664398193, -3.4837918281555176, 1.1300876140594482, -0.9493584036827087, 1.0514436960220337, 1.6370141506195068, 0.9633772969245911, -0.4067528247833252, 0.2874712646007538, 0.5823663473129272, 0.3344678580760956, 0.3498518764972687, 2.3095381259918213, -1.1190296411514282, 1.1333953142166138, 3.354170322418213, -1.2036619186401367, 0.1390213668346405, -0.19769178330898285, -1.3437706232070923, -1.4492169618606567, 0.6725515127182007, -0.680165708065033, 0.6390242576599121, 1.6087013483047485, -1.4153732061386108, -2.670862913131714, 1.6330453157424927, 0.6319447755813599, 1.0357447862625122, 0.10982955247163773, 0.6607415080070496, 0.6322636604309082, -0.721529483795166, -0.4863400459289551, -0.3653552532196045, -0.6761910319328308, -0.6354882717132568], "sim": 0.5653594732284546, "freq": 1208.0, "x_tsne": 120.7471694946289, "y_tsne": 2.571880578994751}]}}, {"mode": "vega-lite"});
</script>


```python
chart_sims.save(f'../out/map-sem-space_{lex}_sims.pdf')
chart_sims.save(f'../out/map-sem-space_{lex}_sims.html')
```

#### differences in neighbours

```python
nbs_vecs = dim_red_nbs_vecs(nbs_vecs, perplexity=70)
```

    /Users/quirin/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:982: FutureWarning: The PCA initialization in TSNE will change to have the standard deviation of PC1 equal to 1e-4 in 1.2. This will ensure better convergence.
      warnings.warn(


```python
nbs_diff = nbs_vecs.drop_duplicates(subset='lex', keep=False)
nbs_diff = (nbs_diff
	.groupby('subreddit')
	.apply(lambda df: df.nlargest(20, 'sim'))
	.reset_index(drop=True)
)
```

```python
chart_diffs = (alt.Chart(nbs_diff).mark_text().encode(
		x='x_tsne:Q',
		y='y_tsne:Q',
		text='lex:N',
		color='subreddit:N',
		# column='subr_nb:N',
	)).interactive()


chart_diffs
```



<div id="altair-viz-d507c4db3af241678d3d9091eff93a0e"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-d507c4db3af241678d3d9091eff93a0e") {
      outputDiv = document.getElementById("altair-viz-d507c4db3af241678d3d9091eff93a0e");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-40f4e27cf6e36f85108bc8f5b2c8f278"}, "mark": "text", "encoding": {"color": {"field": "subreddit", "type": "nominal"}, "text": {"field": "lex", "type": "nominal"}, "x": {"field": "x_tsne", "type": "quantitative"}, "y": {"field": "y_tsne", "type": "quantitative"}}, "selection": {"selector010": {"type": "interval", "bind": "scales", "encodings": ["x", "y"]}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-40f4e27cf6e36f85108bc8f5b2c8f278": [{"lex": "candidates", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.06845252960920334, 1.1415388584136963, 0.8804556727409363, -1.142566204071045, -0.6224484443664551, 3.167255401611328, -0.3299337923526764, 0.07740034908056259, -1.583595871925354, -0.32463204860687256, 1.6747392416000366, 1.9695136547088623, 1.5009632110595703, 0.05019319802522659, 0.8527132272720337, 0.1652047336101532, -1.2427717447280884, -2.9502928256988525, 1.6092967987060547, 3.1317977905273438, 1.5166634321212769, -0.7586343884468079, 0.34843429923057556, -1.0178868770599365, -1.0099067687988281, -1.5964722633361816, -0.6681694388389587, 0.6029216647148132, -0.6958221197128296, -1.8622268438339233, 0.3767610490322113, -1.3271584510803223, 0.932833731174469, -3.675220489501953, 0.5502549409866333, 0.7762888669967651, -2.9422097206115723, 1.8039335012435913, -2.582118034362793, -0.29636913537979126, -3.2295594215393066, 1.269577145576477, -0.0929611325263977, 0.027729207649827003, 3.0487585067749023, 2.0613601207733154, 2.2325282096862793, -0.9274634122848511, -2.3413479328155518, -1.2838472127914429, -0.7945850491523743, 0.8788118362426758, 0.8866395354270935, 1.6620442867279053, -1.597943663597107, -1.0057085752487183, 1.248117208480835, 0.15637175738811493, 0.6091134548187256, -3.5002119541168213, -0.3857225477695465, -0.029191436246037483, -0.6400374174118042, 1.555142879486084, -1.2823171615600586, 1.8320415019989014, 1.5878102779388428, -0.7892467975616455, -1.1368985176086426, -1.1356797218322754, 1.385036587715149, 0.910819411277771, -1.140571117401123, -2.4422519207000732, -0.40274348855018616, -1.549588680267334, 0.10950784385204315, 1.57212495803833, -0.8102763295173645, 1.861938714981079, 2.9245810508728027, 0.6183309555053711, -4.467087745666504, 1.5037981271743774, -2.5673274993896484, -1.150818943977356, -0.2086210697889328, 0.7486494183540344, 1.4432319402694702, -0.052109409123659134, -2.483119249343872, -0.36185595393180847, 2.1526331901550293, 0.0256052128970623, -1.4370701313018799, -0.9475188851356506, -0.9784247875213623, -0.5754059553146362, 2.442857027053833, -0.5109678506851196, -0.8286342024803162, 0.9306701421737671, 1.1543288230895996, -0.37269628047943115, -1.9377447366714478, 2.0513997077941895, -0.30131271481513977, 0.3355407416820526, 2.6284265518188477, -3.7320518493652344, 0.8676887154579163, -1.0214245319366455, 0.7306825518608093, -2.6152830123901367, 0.756365180015564, -1.6023352146148682, 2.2856359481811523, 0.007956569083034992, -0.7235228419303894, -2.7240376472473145, -2.1884968280792236, 1.2781965732574463, -0.35816848278045654, -2.919647216796875, 0.832656741142273, 5.3676676750183105, -2.5018439292907715, 0.15723705291748047, 1.17588210105896, -0.9364197850227356, -1.0017337799072266, 0.21196401119232178, -4.690650939941406, -1.6883225440979004, 1.6399049758911133, 0.5995707511901855, -0.5463524460792542, -2.909010171890259, 0.08116518706083298, -0.9338349103927612, 5.057669162750244, -0.9898715019226074, 0.032845400273799896, -1.5810940265655518, -0.7237266302108765, -1.290727138519287, -1.1293877363204956, -0.5660204887390137, 1.3745806217193604, -1.6913634538650513, 1.5506272315979004, 0.061756305396556854, 1.0091278553009033, 0.1818670928478241, 4.048049449920654, -0.11152373254299164, 2.8943159580230713, 0.21163326501846313, -0.8676732778549194, -1.6788594722747803, -0.7496339082717896, 4.331779479980469, 1.3595120906829834, 1.890046238899231, 1.5138550996780396, 2.1984450817108154, -0.8633642792701721, -0.4668896496295929, 0.3384958505630493, 0.885133683681488, -1.7375152111053467, -1.5514013767242432, -0.6286267042160034, 1.4316678047180176, -0.4409816861152649, -2.1081128120422363, -0.3966287076473236, 1.1584265232086182, -0.9089121222496033, -1.8727266788482666, -1.204331874847412, 2.3584072589874268, -2.595177173614502, -0.6238438487052917, -2.449601173400879, 0.8050606846809387, 0.9748712778091431, -1.8573791980743408, 0.2530331015586853, -2.19069766998291, -1.8102091550827026, 0.48158925771713257, 0.13850346207618713, -1.1724395751953125, -0.7067437171936035, -3.522475004196167, -1.0569393634796143, 0.8504513502120972, 0.7676689028739929, 0.8858399391174316, -4.076595306396484, 1.9842382669448853, -0.5702388882637024, 0.20208093523979187, -0.945242166519165, -2.1274168491363525, -2.8668272495269775, 3.6277077198028564, -1.4688810110092163, 1.3255223035812378, -1.1866060495376587, -0.687542200088501, -1.49526846408844, -0.6750629544258118, 2.750946283340454, 0.3661094903945923, -1.7008800506591797, 0.029865646734833717, 1.192836046218872, -0.2295132577419281, -0.05525878816843033, 1.7167305946350098, -2.192483425140381, -2.4659454822540283, 1.2456636428833008, -0.20360346138477325, 0.9131380915641785, 1.528957724571228, 1.3027336597442627, -2.428126335144043, -0.29559141397476196, 0.16349829733371735, -1.4282424449920654, -0.41527798771858215, 1.7299854755401611, -2.280627489089966, 0.9492124319076538, 5.850632667541504, -3.4335849285125732, -1.948125958442688, -2.082648992538452, -2.634706497192383, 1.4429714679718018, -1.2583885192871094, -0.17413870990276337, -0.5892731547355652, -0.1405894011259079, 0.9652339220046997, 0.6062386631965637, 0.8875123262405396, 0.8567684292793274, -1.3124597072601318, 2.338820219039917, -0.1624145358800888, -1.7110295295715332, 1.9095349311828613, 1.623496413230896, -0.3910505473613739, 1.658247470855713, -1.7249467372894287, 0.4675557017326355, -0.29292941093444824, -2.5948009490966797, 1.4994127750396729, 1.1495835781097412, -0.60566246509552, 1.6304186582565308, -0.657894492149353, -0.4790826737880707, 0.4411816895008087, 0.8107678890228271, -1.9953943490982056, 1.1009025573730469, -2.141955852508545, 1.396376132965088, 2.1583359241485596, -1.5465155839920044, 1.775546908378601, -2.6354262828826904, -1.408606767654419, 1.007779598236084, 0.8777630925178528, -2.030611753463745, 1.39594304561615, -0.36170390248298645, -1.0883455276489258, -0.8950857520103455, -3.0718345642089844, 0.3537704050540924, 1.3020994663238525, -0.6436963081359863, -0.8795176148414612, 2.2560033798217773, 0.8216496109962463, 0.864078938961029, -2.319356679916382, -2.7154455184936523, -1.3044766187667847, 0.19619840383529663, -0.5186092853546143], "sim": 0.5222594141960144, "freq": 4842.0, "x_tsne": 25.19969940185547, "y_tsne": -0.06266182661056519}, {"lex": "commercially", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.9494524002075195, -0.3043292760848999, -1.472187876701355, -0.6493716239929199, -0.439471036195755, 1.1966583728790283, -1.4950051307678223, -0.6863470077514648, 0.3068132698535919, 0.3864716589450836, 1.8772180080413818, 0.9553171396255493, 1.8857123851776123, 0.13032259047031403, 0.26260697841644287, -1.0736324787139893, -0.43507662415504456, -1.2709131240844727, -0.37578126788139343, -0.8627557158470154, -1.700440764427185, -0.8085081577301025, -0.18890389800071716, 1.4426040649414062, 0.7090155482292175, -0.26290953159332275, 0.034087710082530975, -0.3899548053741455, 0.23765656352043152, -0.01995536871254444, 1.1677799224853516, 0.7440906167030334, 0.5736686587333679, 0.48555147647857666, 0.5340729355812073, 1.1662471294403076, -0.31101882457733154, 0.6736233830451965, 0.5621968507766724, -0.5130717754364014, -0.24612078070640564, 0.6931391954421997, -0.22140289843082428, 0.03134220466017723, 0.7137058973312378, -0.19336345791816711, -1.1292767524719238, 0.8070483207702637, -0.2566237151622772, -0.8030185699462891, -0.7214489579200745, 1.5048948526382446, 0.036815743893384933, 0.3738935589790344, 0.09499470889568329, 0.06128860265016556, 0.43183210492134094, 0.7002384066581726, 0.9949386119842529, -2.0472817420959473, 0.8509140014648438, 1.0072892904281616, 1.454912543296814, 0.7850199937820435, 0.5341346263885498, -0.22409914433956146, -0.8356530070304871, -1.6131740808486938, -0.40495532751083374, -0.27251189947128296, 1.1937294006347656, 0.8525068759918213, 0.41801199316978455, -0.11751407384872437, -0.5805870294570923, -1.179092288017273, -0.7162289023399353, -0.026723813265562057, -0.367717444896698, 0.20331920683383942, 0.3775178790092468, -1.2231953144073486, -0.6612453460693359, -0.300357848405838, -1.7360576391220093, 0.3671368360519409, -0.5723521113395691, -1.1735913753509521, 0.4989275336265564, 1.3684006929397583, 0.1013273075222969, -0.13258637487888336, 1.4208284616470337, 0.26749345660209656, -0.6655280590057373, -0.2315751612186432, 0.2674466073513031, -0.5885264873504639, 0.3264450132846832, 0.34328341484069824, 0.6818708181381226, 0.28172069787979126, 0.034496892243623734, 0.31284791231155396, -1.5280965566635132, -0.4823921024799347, 0.7741459608078003, -1.5797357559204102, -0.7314302921295166, -1.0611441135406494, -0.0721387043595314, -1.110185146331787, -0.09694651514291763, -0.5692533850669861, -0.09873891621828079, -0.13672956824302673, 0.46029818058013916, 0.9582273364067078, 0.22834178805351257, -0.8789016008377075, 1.459652066230774, -0.3181614875793457, 0.2243259698152542, 0.04620958864688873, -0.5758932828903198, 2.6807382106781006, -0.34913089871406555, -1.66227388381958, 0.0574583075940609, -1.042658805847168, 1.0445455312728882, 0.4631566107273102, 0.37361183762550354, 0.07660839706659317, 1.0637370347976685, 1.297546148300171, -0.21812191605567932, -0.7783690690994263, -0.7136362791061401, 0.3763289153575897, 0.02311294712126255, -0.4696534276008606, -0.39424002170562744, -0.18307381868362427, -1.4564695358276367, -1.2948520183563232, -0.9492285847663879, -0.24310772120952606, -0.6007077097892761, -0.25234362483024597, 0.9630622863769531, -0.6735199093818665, 1.3029789924621582, 0.35716527700424194, -1.2177551984786987, -1.7748241424560547, 0.016216468065977097, -1.0924075841903687, -1.6819607019424438, 0.024622375145554543, 0.9381402730941772, -0.5758785009384155, -0.06426169723272324, -0.275246798992157, 0.22332417964935303, 0.4446810185909271, 0.3151089549064636, -0.18538451194763184, -0.5865154266357422, -0.26258742809295654, 0.9665311574935913, -1.236372947692871, -0.6291753053665161, 0.9326640963554382, -0.6662378907203674, -1.4536186456680298, -0.2730238437652588, 0.6228556632995605, 1.7651926279067993, 0.41348710656166077, 0.5509727597236633, 2.0268726348876953, -0.2159997969865799, -0.006504778750240803, -0.5105426907539368, -0.6409258246421814, 1.7856303453445435, -0.0015877713449299335, 1.1757258176803589, -1.2376608848571777, -1.2804299592971802, -0.8362532258033752, -0.6716691255569458, 0.5590037703514099, -0.09718301147222519, -0.14696447551250458, -0.46869784593582153, -0.35188060998916626, -0.3903319239616394, -0.9450567364692688, 0.01213243417441845, -0.024882722645998, -0.626697838306427, 1.2766129970550537, 0.9420320391654968, 0.34602272510528564, -1.1322859525680542, 0.964873194694519, 2.141493320465088, 0.8566065430641174, -1.2037862539291382, -1.0791282653808594, -0.44406092166900635, 0.33211424946784973, 1.867377519607544, 0.49610674381256104, -1.9989031553268433, -1.426293969154358, -1.1605521440505981, -1.2589731216430664, 0.5138412117958069, 2.383817434310913, -0.6514228582382202, -1.2067629098892212, -0.29881852865219116, -0.5317818522453308, 1.0367302894592285, 1.1088082790374756, 0.6958203911781311, -0.11751914769411087, -1.2293317317962646, -0.294752836227417, -0.16864895820617676, -1.4594359397888184, 0.8141815066337585, 0.8374451994895935, -0.00889628753066063, 2.1423118114471436, 0.9726313948631287, -1.3699240684509277, -0.8974327445030212, -1.0689396858215332, -0.3901897668838501, 0.11377814412117004, -0.7067109942436218, -0.024082206189632416, 1.5026941299438477, 2.04105544090271, -1.011457085609436, 1.1837266683578491, -0.9304275512695312, -0.5145289897918701, 0.8557071685791016, 0.1747829169034958, -0.17488934099674225, -0.09208475798368454, -1.1289554834365845, -1.0281864404678345, 0.8784895539283752, -2.0263071060180664, -0.02973824180662632, 2.051816701889038, -1.006189227104187, -0.5800844430923462, 0.1178402304649353, 0.9646778702735901, 1.6418031454086304, -0.44149085879325867, 0.11996103823184967, -0.1824284940958023, -0.39369872212409973, -0.013374836184084415, -1.7731313705444336, 1.4192895889282227, 1.154143214225769, 0.8081423044204712, -1.1147425174713135, 0.9342296719551086, 0.1861838847398758, -0.7622779011726379, -0.02847684919834137, -0.08336023986339569, 0.2778324782848358, 0.12995314598083496, 0.4470760226249695, 1.663563847541809, 0.5558258295059204, -0.9442930221557617, -1.6115983724594116, 0.06308609992265701, -0.2872769236564636, 0.3077617883682251, -1.3428544998168945, 0.6387491822242737, 1.0638076066970825, -0.7073197960853577, -1.0400274991989136, 0.602711021900177, -0.05552306026220322, 0.4373264014720917], "sim": 0.45659637451171875, "freq": 150.0, "x_tsne": 9.193573951721191, "y_tsne": 5.498041152954102}, {"lex": "strands", "type": "nb", "subreddit": "Coronavirus", "vec": [1.7250185012817383, -0.016960112378001213, -0.29245996475219727, -1.465551733970642, 0.11692763864994049, -0.1489177793264389, 0.8603444695472717, -0.6687768697738647, -0.008629463613033295, -1.3577207326889038, 0.6132813692092896, -0.7479682564735413, 1.6585586071014404, -0.7336528897285461, 0.04622606188058853, 0.3917190432548523, -1.109531044960022, 0.04832005128264427, -0.14428159594535828, -1.2079691886901855, -0.7562205791473389, -0.7833665013313293, -0.7676770091056824, -1.0395152568817139, 2.5663769245147705, 0.27899032831192017, -2.2666690349578857, -0.32857009768486023, -1.5492501258850098, -1.7215250730514526, -1.2016432285308838, 0.5408412218093872, -0.6668205857276917, -0.6346274018287659, 1.1446971893310547, 1.0110923051834106, -0.4475055932998657, 0.11199596524238586, -1.9712021350860596, -1.990662693977356, -0.12237781286239624, -0.5113401412963867, 1.299431562423706, -1.9901223182678223, 0.7014092803001404, 0.01112364325672388, 0.7510348558425903, 1.3386836051940918, -1.6607799530029297, -0.7469752430915833, -1.3519901037216187, 0.4361816346645355, 0.1385430097579956, -0.0739913284778595, -1.2477389574050903, 1.7511781454086304, 0.1298876851797104, -0.6162905693054199, 2.262782573699951, -2.7908291816711426, 0.9805753231048584, 0.16402797400951385, 0.07386106997728348, -0.13977432250976562, -2.3845059871673584, 1.272734522819519, 1.393708348274231, -1.6808923482894897, -1.5154060125350952, -0.2432723194360733, 1.9980430603027344, 0.30497822165489197, -0.17688912153244019, -0.591492772102356, 1.0396562814712524, -1.882421612739563, 0.9233777523040771, 0.7512038946151733, -1.6736143827438354, 2.155271291732788, 0.9896687269210815, -1.4286662340164185, -1.6440240144729614, 1.6665362119674683, -1.2269586324691772, -0.8970980048179626, -1.451680064201355, -0.08061248809099197, -0.15259966254234314, 0.6339675188064575, -0.49644148349761963, -0.6275217533111572, 0.3316332995891571, 0.8721213340759277, -1.4989774227142334, -0.41280898451805115, -1.7233667373657227, -0.3479279577732086, -0.24880483746528625, -1.1290258169174194, -0.40638720989227295, 0.8154776096343994, -0.25739631056785583, -0.2849223017692566, -1.207584261894226, -0.256511926651001, 0.05783920735120773, -0.053690675646066666, 4.198399066925049, -0.5810428261756897, 2.236950397491455, -1.4723668098449707, 0.3817577362060547, -0.9328594207763672, 0.04426717385649681, 0.6099150776863098, -0.11418336629867554, 0.25253477692604065, 0.2622023820877075, 1.2868657112121582, -0.3139013946056366, -0.8905723094940186, -1.5180490016937256, -2.218324899673462, 1.3384151458740234, 2.570683479309082, 1.3801229000091553, 1.3695337772369385, 0.31915009021759033, 0.7015663385391235, 1.5649157762527466, -1.6530007123947144, -2.443328619003296, -1.9997628927230835, -0.6051817536354065, 0.9749528765678406, 0.1603291779756546, -0.5934618711471558, 2.3152942657470703, -0.5744971036911011, -0.04563012719154358, -1.8495854139328003, 0.03913518786430359, -1.6500962972640991, -0.9376188516616821, 0.7904970645904541, -1.708882451057434, -3.5505332946777344, 1.2211045026779175, -0.4541998505592346, -0.43696436285972595, -0.7914489507675171, 2.1585445404052734, -0.10273716598749161, -0.15244649350643158, 0.07746288180351257, 2.6355323791503906, -0.5899924635887146, -0.5438581109046936, -0.5447129607200623, 0.05532503128051758, -0.3249267637729645, -0.2027347981929779, 2.8274331092834473, 0.6452198624610901, -0.1498298943042755, -1.486359715461731, -0.3373000919818878, 0.9701481461524963, -0.9804309606552124, -0.6751901507377625, -0.33868643641471863, -0.45759692788124084, 0.01604353077709675, 0.31535762548446655, -0.8918825387954712, -1.7383795976638794, -1.2387689352035522, 0.5512285828590393, -0.44893625378608704, -1.5355007648468018, 0.5771965384483337, 0.08874842524528503, 0.6047869324684143, -1.5679882764816284, -0.06291379779577255, 1.119913935661316, -1.9546782970428467, 0.7947730422019958, -2.108058452606201, -1.5398892164230347, -0.5945018529891968, -1.2865009307861328, 1.8822273015975952, -1.7556276321411133, -1.4087061882019043, 1.8570326566696167, 0.484536737203598, 0.6364670991897583, -0.2048467993736267, -0.694071352481842, 0.11856254935264587, -1.9220068454742432, 1.4737814664840698, -0.6550919413566589, 0.3630007803440094, -3.447197675704956, 2.9031662940979004, -1.2377593517303467, 0.03761911019682884, -0.905635416507721, 0.17896173894405365, 1.302786111831665, -0.23075269162654877, 0.47239330410957336, -0.40352728962898254, -1.4010573625564575, 0.05622274801135063, 0.3679213523864746, -0.3975963294506073, 0.7150813937187195, 2.06781005859375, 1.7818292379379272, 0.13120095431804657, -1.5239667892456055, 1.065195083618164, 1.0466487407684326, 1.3009870052337646, 0.09489229321479797, 0.007680518086999655, -0.006871914491057396, -0.4934752285480499, -1.1778124570846558, 0.4513233006000519, 0.6026726365089417, -0.9290200471878052, 0.08109921216964722, 1.1162093877792358, -1.7537163496017456, -0.9628866910934448, 0.16115276515483856, -1.5273041725158691, 0.14231343567371368, -2.146805763244629, -0.09126822650432587, -1.627913475036621, 2.3230953216552734, 0.7127881646156311, -0.14774756133556366, 2.5161545276641846, -0.18513788282871246, -0.40770527720451355, 0.7033522725105286, 1.6165084838867188, -0.6051506400108337, -0.43726563453674316, -0.8891693949699402, -2.261381149291992, -0.1365812122821808, -2.089512348175049, 1.706884741783142, -0.19631966948509216, -0.39094147086143494, 0.19232453405857086, 1.6699426174163818, -0.6701852679252625, 0.5208936333656311, 0.21349558234214783, -1.5888441801071167, 1.8098148107528687, -0.19108420610427856, 0.4886179566383362, -1.3379223346710205, -2.60510516166687, 1.005391001701355, 1.4037213325500488, -0.3766007721424103, 3.0664734840393066, -1.5133267641067505, 0.6084038019180298, 1.504326581954956, 0.559778094291687, -0.6897495985031128, 1.3169573545455933, 0.6590036749839783, 0.1392829716205597, -0.5778947472572327, -1.8522988557815552, 0.06768374890089035, 0.23272311687469482, 0.7365631461143494, -0.27426770329475403, 0.6552690267562866, -0.9459587335586548, 1.5429953336715698, -0.0658130943775177, 0.31764546036720276, -0.06317079812288284, -0.7518040537834167, -0.3286004364490509], "sim": 0.44637003540992737, "freq": 182.0, "x_tsne": 2.546952962875366, "y_tsne": 20.948986053466797}, {"lex": "adenoviruses", "type": "nb", "subreddit": "Coronavirus", "vec": [0.4742138683795929, 0.31418514251708984, 0.873507022857666, -0.848550021648407, 0.3055362403392792, -0.028107961639761925, -0.175106480717659, -0.7623668909072876, 0.6057085990905762, -0.6782325506210327, -0.39443162083625793, -1.1900858879089355, -0.018964724615216255, -0.7145906686782837, 1.5562676191329956, -1.1340713500976562, -1.1106319427490234, -1.0990492105484009, 0.24008619785308838, -0.02759038470685482, -0.5172186493873596, 0.08431953191757202, -0.3782273828983307, 0.6322763562202454, 0.7384622097015381, 0.13817238807678223, -0.3529881536960602, -0.153628870844841, -0.39774709939956665, -0.17226864397525787, -0.17815794050693512, -0.30440762639045715, -0.6225956082344055, -0.46880438923835754, 1.250100016593933, 0.9056714773178101, -0.5422272682189941, -0.3177388906478882, -1.2639905214309692, -0.471477746963501, -0.2293919175863266, -0.937384307384491, -0.763888955116272, -0.07330861687660217, 0.9650344848632812, 0.4584770202636719, 1.0564764738082886, 0.7114297151565552, 0.33258092403411865, 0.3580528795719147, -0.37750178575515747, -0.4539160430431366, -1.0974388122558594, -0.11599721759557724, -0.7239758968353271, -0.502487063407898, 0.38795724511146545, 0.03239750489592552, 0.374000608921051, -2.4328010082244873, 0.32963329553604126, -0.06803659349679947, 0.526608407497406, 0.49736860394477844, -0.9434443116188049, 1.5792425870895386, 0.7988944053649902, -0.22606812417507172, -1.4455724954605103, -0.4628700315952301, 1.254624843597412, -0.3490484058856964, -0.583411693572998, 0.8139962553977966, -0.7142556309700012, -1.19866943359375, 0.47098100185394287, -0.3253328800201416, -0.6727171540260315, 0.8224214911460876, 0.07232654839754105, 0.009667004458606243, -1.0234675407409668, -1.0135302543640137, -0.04646258428692818, 0.1929941028356552, -0.00953256618231535, -0.0962081179022789, 0.6313483715057373, -0.012890706770122051, 0.06615252792835236, -0.4282780587673187, 1.2056922912597656, 0.587897539138794, -0.639897346496582, -0.33726996183395386, 1.6028311252593994, 0.8333760499954224, 0.7754411697387695, -0.06792185455560684, -0.5720955729484558, -1.0585325956344604, 1.0700839757919312, 0.6273607015609741, -0.7896710634231567, 0.7469267249107361, -1.0570337772369385, -0.48147913813591003, 0.20756642520427704, 0.07005809992551804, -0.4703473746776581, -0.09979449957609177, -1.2543357610702515, -1.0319045782089233, -0.5950215458869934, -0.21770597994327545, -0.04037492722272873, -0.6298325657844543, 0.34534162282943726, -0.6972880959510803, 0.007393761072307825, 1.429818034172058, -0.5933727025985718, -0.40627366304397583, 0.8629809617996216, 1.9511377811431885, -0.3813721835613251, 0.8916347622871399, 0.1362450122833252, 1.5904936790466309, -0.2037157416343689, -0.9341447353363037, -0.6232736110687256, -1.1125800609588623, 0.8494714498519897, 0.3644688129425049, -0.3038186728954315, -0.8565817475318909, -0.8546452522277832, -1.2597084045410156, 0.041298799216747284, -0.2968001067638397, -0.39348697662353516, -0.1385526955127716, 0.3877842426300049, -0.36152878403663635, 0.43374931812286377, -1.0165046453475952, 0.7403594255447388, 0.41653692722320557, 1.265649437904358, -0.07609324157238007, 1.2512131929397583, 0.4627552926540375, 0.6903342604637146, 0.651516318321228, -0.5181664228439331, -0.6864873766899109, -1.3478227853775024, 0.21375855803489685, -0.30631911754608154, 0.9275044202804565, -0.7414767146110535, -0.769569456577301, 0.28910142183303833, 0.188578262925148, -0.5894633531570435, -0.0713043212890625, 0.3751677870750427, 0.8980376720428467, -0.7098373770713806, 0.7358011603355408, -1.4107091426849365, 0.48966360092163086, 0.22546996176242828, -1.7044548988342285, 0.3055666387081146, -0.5349614024162292, -0.7722750306129456, -0.6603289842605591, -0.15228301286697388, 0.06390129029750824, 1.0792953968048096, -1.0598196983337402, 0.06772013008594513, -0.9876536130905151, 1.118759036064148, -0.6855838298797607, 0.899553120136261, -0.7351870536804199, 0.46944674849510193, -0.48290562629699707, -1.3188034296035767, -0.4149024784564972, 0.015356797724962234, -1.0804723501205444, 0.7741956114768982, 0.5770434737205505, 0.2289789766073227, -0.3720468580722809, -0.8840584754943848, 0.5727789402008057, -1.192969560623169, 0.2827155590057373, -0.07084503769874573, 0.47607243061065674, -1.1859463453292847, 0.3174210786819458, 0.3498944640159607, 0.2666330635547638, -1.249218225479126, -0.08311571925878525, -0.0023827441036701202, 0.26225024461746216, 0.6948208808898926, -0.5078374743461609, -1.1766881942749023, -1.9115062952041626, -0.8506482839584351, -0.2256254106760025, 1.1135241985321045, 1.0298956632614136, 1.2904562950134277, -0.30889782309532166, 0.09735369682312012, 0.5684138536453247, 0.33898699283599854, 2.2555644512176514, -0.26112574338912964, -0.8054639101028442, -0.5963431596755981, -0.7282658815383911, -0.3733507990837097, -0.19506697356700897, 0.34444791078567505, -0.05851898342370987, -0.46858516335487366, 0.8835016489028931, 0.017699170857667923, -0.6375479698181152, 0.22198203206062317, -1.7155659198760986, -0.12722232937812805, -0.9930040836334229, 0.41652020812034607, -0.7382382750511169, 0.6296277642250061, 0.7553483247756958, 0.5612024068832397, 1.4532241821289062, -0.8939400315284729, -1.0976983308792114, 2.112976551055908, 0.3943687081336975, -0.33653730154037476, -0.2902269661426544, -0.21428461372852325, -1.670695185661316, 0.42679545283317566, 0.4423491060733795, 0.7303162217140198, -1.2739938497543335, -0.6082291007041931, -1.1996448040008545, -0.8131090998649597, -0.5817967653274536, -0.031361646950244904, -0.44798576831817627, -0.8657890558242798, -0.041884422302246094, -0.6665341854095459, -0.8921508193016052, -0.8736797571182251, -0.7968315482139587, 0.7302424907684326, 0.8471881747245789, -0.40709248185157776, 1.9486559629440308, -0.24452614784240723, -0.36317431926727295, 0.6701669692993164, -0.005231006536632776, -0.21909783780574799, 0.672981858253479, -1.1286473274230957, 1.184332251548767, 0.14863374829292297, -1.4848806858062744, -0.45666536688804626, -0.6094859838485718, 1.0698860883712769, 0.5335790514945984, 1.2421913146972656, -0.9336288571357727, 0.6014873385429382, -0.02933250367641449, -0.46131810545921326, -0.33995142579078674, -0.43337684869766235, 0.13117751479148865], "sim": 0.4208108186721802, "freq": 41.0, "x_tsne": -0.3911755084991455, "y_tsne": 25.752077102661133}, {"lex": "batches", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.5931841731071472, 1.00252103805542, 0.04529110714793205, -0.9296517968177795, -0.8155503273010254, 1.6041845083236694, 0.7013376951217651, -2.287504196166992, 1.622268795967102, -0.41200074553489685, -0.93455570936203, 0.8105391263961792, 1.1987208127975464, -0.9137882590293884, 1.9197700023651123, 0.7095745205879211, -1.270886778831482, -2.079343557357788, -2.261765718460083, -1.0066932439804077, -0.5336138606071472, 0.9409521222114563, -1.9640164375305176, 0.0665171816945076, 2.9379289150238037, -2.5925140380859375, -2.1548593044281006, -1.6123027801513672, 0.69011390209198, -1.886278510093689, -1.534852147102356, 0.4873814582824707, 0.5651730298995972, -3.2780227661132812, 2.181178092956543, 0.9377812147140503, 0.181168794631958, -1.0526795387268066, -0.8143370151519775, -0.006048909854143858, -0.7154902219772339, 1.1959627866744995, -0.017195604741573334, 0.5236889123916626, -1.261619210243225, 1.2228195667266846, 0.720679521560669, 0.6997856497764587, -0.38475915789604187, 0.568609356880188, -1.776686429977417, 1.3962976932525635, -2.4365007877349854, 0.7300257682800293, -1.1928690671920776, 2.468055009841919, -0.19574227929115295, -1.7890387773513794, -2.2177510261535645, -1.0068367719650269, -1.4571317434310913, 1.1574742794036865, 1.1911207437515259, 0.5665447115898132, 0.789688229560852, 0.8344784379005432, 1.8965588808059692, -0.6836032271385193, -0.6319516897201538, 0.9425073862075806, 1.5488694906234741, 2.5088257789611816, -1.2220213413238525, -0.43430569767951965, -1.8000115156173706, -1.2689417600631714, 3.1212759017944336, 1.729673981666565, 0.022212110459804535, 0.7843701243400574, 1.4355496168136597, -1.0329231023788452, -0.09931415319442749, 0.5063744187355042, -1.5810465812683105, -2.2857582569122314, 1.2070561647415161, -0.3645685911178589, 1.1631782054901123, 0.26566654443740845, -1.3281737565994263, -0.8011219501495361, 0.6169417500495911, 0.5457867980003357, -0.307504266500473, -0.871679425239563, -2.3293204307556152, 0.8763354420661926, 0.732621967792511, 0.9072442054748535, -0.6884927749633789, 1.1017179489135742, -1.4127388000488281, -0.8325837254524231, -2.36430287361145, -1.882925033569336, 1.721444845199585, -0.91875821352005, 1.991866946220398, -0.3833177387714386, -0.9606494903564453, -1.5529495477676392, -0.19834081828594208, 0.6316691040992737, -0.4890706539154053, 0.5523797273635864, -0.6042866706848145, -0.2842528223991394, 0.1220213770866394, 1.9495892524719238, -0.8112826347351074, 0.5826659202575684, 0.7203831672668457, -0.14770591259002686, -0.034365590661764145, 3.747032880783081, -1.068406343460083, 0.05202464386820793, -0.774183452129364, -1.0603117942810059, 0.6270852088928223, -0.1393645703792572, -3.077738046646118, -0.7868994474411011, 0.5847910046577454, -0.12763261795043945, -1.871740698814392, -0.7647906541824341, -0.7995057702064514, -0.10357682406902313, 1.7095307111740112, -0.9689918160438538, 0.4202503263950348, 0.5400812029838562, -0.8055732250213623, -0.07687679678201675, -2.384803056716919, -0.5969994068145752, 1.049647331237793, -1.4789866209030151, 0.44943544268608093, -0.23262226581573486, 0.4821929931640625, 0.2825480103492737, -1.8457727432250977, 0.7241519689559937, 0.7416343688964844, -0.08414874225854874, -1.602299451828003, 0.054676495492458344, 0.05947715789079666, -1.3191419839859009, 0.18254338204860687, 2.712797164916992, 0.591858446598053, 2.9056484699249268, 0.351771742105484, -0.3861364424228668, -0.47382840514183044, 2.8865482807159424, 1.5940067768096924, -0.4906923174858093, -3.1286139488220215, -0.20368142426013947, -0.13329149782657623, 0.6176759600639343, -2.1502573490142822, -1.2122026681900024, 1.4893501996994019, -0.9298898577690125, -0.2601255178451538, 0.9327592849731445, -0.23848877847194672, 0.757515549659729, -1.9516834020614624, -0.33858439326286316, 0.35748085379600525, -1.024074673652649, -0.0933678075671196, -1.9638915061950684, -0.827815592288971, 0.6149697303771973, 0.6760889887809753, 2.081629514694214, -2.0578083992004395, -0.5112103819847107, -0.27741411328315735, -1.864251971244812, 1.8563464879989624, -2.457319974899292, -1.3455288410186768, 0.7571740746498108, -0.17219790816307068, 0.8136405944824219, -0.35726574063301086, 0.4205903112888336, -1.0541757345199585, 1.5433768033981323, 0.501528263092041, 0.4972473680973053, 0.7460274696350098, -1.6303783655166626, 1.0660486221313477, -0.09482336789369583, 0.9008394479751587, 1.277510166168213, -1.2076451778411865, -1.5773183107376099, 0.7077027559280396, 0.6447091698646545, -0.05921756848692894, 1.0191493034362793, 1.2904165983200073, -0.9781333208084106, -0.0656263679265976, 0.5597995519638062, 0.3645503520965576, 0.7915488481521606, 1.4440689086914062, 0.6185271143913269, -0.24017496407032013, -2.096039295196533, -4.185266494750977, 0.6696251034736633, 3.67032790184021, -1.2374581098556519, 0.9480554461479187, 4.3080925941467285, -1.3653379678726196, -1.2143781185150146, 1.0049026012420654, 0.9545662999153137, 0.8373715281486511, -2.1423451900482178, -0.15743400156497955, -0.6393195986747742, 2.0635979175567627, 0.8158972263336182, -1.0709463357925415, 0.9331861734390259, 2.32637095451355, -2.8218801021575928, 1.394824504852295, 1.827597975730896, -0.24452367424964905, 0.48118776082992554, 1.6907875537872314, 0.16481883823871613, -1.4434449672698975, -0.33612075448036194, -0.34749677777290344, 2.196272850036621, -0.6369999051094055, -2.024042844772339, 0.24534182250499725, 1.2589459419250488, -0.5413059592247009, -0.38035282492637634, 2.598708391189575, -0.45155414938926697, -1.8151650428771973, 1.1451776027679443, -0.7970988750457764, 0.5356698632240295, 2.3234546184539795, 1.103867769241333, -0.027654776349663734, -0.6706356406211853, 0.8887509107589722, -1.527631402015686, -1.1873823404312134, 0.6640598773956299, 0.6843974590301514, -0.16948114335536957, -0.5940107703208923, -0.6108677983283997, -0.8615586161613464, -2.4230599403381348, -0.707391619682312, -1.2382113933563232, -1.8142590522766113, -0.889427900314331, 1.2610012292861938, 0.7744131684303284, 0.33074891567230225, 0.877366304397583, 0.19948163628578186, 1.1372253894805908, -1.7779452800750732, 0.321213960647583], "sim": 0.41588982939720154, "freq": 302.0, "x_tsne": -0.32642802596092224, "y_tsne": 19.817167282104492}, {"lex": "vials", "type": "nb", "subreddit": "Coronavirus", "vec": [0.7645885944366455, 0.9584415555000305, -0.48908013105392456, -1.2345350980758667, -2.6443722248077393, 0.8336604237556458, 0.9927849173545837, -0.4423106610774994, 0.9962586760520935, -0.7480011582374573, 0.47089332342147827, 0.47112035751342773, 2.4332501888275146, -1.6531662940979004, 0.5422534346580505, 2.3177812099456787, -0.32300978899002075, -0.9334741830825806, 0.8044921159744263, -2.671067476272583, -1.2074216604232788, -0.7627653479576111, -2.8019604682922363, -0.03499238193035126, 2.8566582202911377, -1.878261685371399, 0.8047245740890503, 0.07451356947422028, -1.4198715686798096, -1.1288806200027466, 0.3521127998828888, -1.5972650051116943, 0.9445867538452148, -0.19747187197208405, 1.3974497318267822, 2.603290557861328, 0.35425201058387756, -1.369982361793518, 0.3825569748878479, -1.4871708154678345, 1.00298273563385, 0.7278422713279724, 0.7543311715126038, 0.8700473308563232, -0.04939965531229973, 1.1193996667861938, 2.2356114387512207, 0.5283831357955933, -0.06880570203065872, -0.5563046336174011, -1.9610587358474731, 1.019925594329834, -2.6637866497039795, 0.8772305250167847, 0.22935806214809418, -0.1181388571858406, 1.1446834802627563, 0.16413529217243195, -0.736190140247345, -2.693713665008545, 1.9387459754943848, 3.3825063705444336, -0.6159571409225464, -1.9784594774246216, -1.1034013032913208, 0.17476285994052887, 1.16631019115448, -0.688460111618042, 0.2937004864215851, 1.243504524230957, 2.8108749389648438, 1.9751142263412476, -2.580717086791992, 0.003082267940044403, -1.656024694442749, -1.4094868898391724, 2.181436777114868, 0.5431416034698486, -1.546570897102356, 0.4593636095523834, 0.10643795877695084, 0.2664688527584076, -0.7893930077552795, 3.6295619010925293, -2.5541231632232666, -0.9266311526298523, -1.5441347360610962, 2.0344016551971436, 1.3732450008392334, 1.3497604131698608, -0.34139370918273926, -2.334712028503418, 3.514029026031494, -0.744561493396759, -2.013969659805298, -0.006715989671647549, -0.5305848121643066, -0.3973752558231354, -1.2349798679351807, 0.7373111844062805, -0.4640055298805237, 0.4163680672645569, -0.0019273355137556791, 0.20296935737133026, -2.9867749214172363, -0.8792678713798523, -1.3710464239120483, 0.0772327408194542, 1.634250521659851, 0.3541482985019684, -0.662355363368988, -2.050008773803711, -1.376020073890686, -0.05520617961883545, 0.09873466938734055, 0.34687045216560364, -0.8452884554862976, -0.5489429235458374, 0.5402866005897522, 1.0616681575775146, -0.31062018871307373, 1.2524356842041016, -0.6416436433792114, -2.9162933826446533, -0.40584349632263184, 4.240508556365967, -0.9620295166969299, -0.2833753526210785, 1.116010069847107, -0.23978279531002045, 1.649118423461914, 1.0202492475509644, -3.118595600128174, 0.7231006026268005, -0.5478686094284058, 0.4353565573692322, -2.9083662033081055, -0.9902980327606201, 1.5749460458755493, 0.4181790053844452, 1.892555832862854, -0.36925724148750305, -0.7520599365234375, 0.6684566736221313, -0.26420480012893677, -1.9224357604980469, -4.914751052856445, -0.8129228949546814, -0.22755487263202667, 0.06074449047446251, -0.3572418987751007, -1.177281141281128, 1.8488185405731201, -1.3273992538452148, -1.9217146635055542, -0.48441198468208313, -0.33150362968444824, -0.4387594759464264, -1.1121100187301636, -0.2488347589969635, 1.6267940998077393, -0.8521052598953247, -0.8151057362556458, 1.6469919681549072, -0.006582397501915693, -0.48157915472984314, 0.055930230766534805, 1.6358171701431274, -1.1254222393035889, 4.140347957611084, 1.3364543914794922, -0.3262540400028229, -1.1122019290924072, 1.4440228939056396, -1.0585074424743652, -0.14721237123012543, -0.8091359734535217, -0.804129958152771, 2.620875597000122, 1.6104837656021118, -0.5402622818946838, 2.2926955223083496, 0.008848171681165695, 0.8732686638832092, -2.6315741539001465, 0.20318137109279633, 1.1330416202545166, -0.5901361107826233, 0.7225160598754883, -2.0093891620635986, -1.2187738418579102, -1.5414797067642212, 0.3580579459667206, -1.604427456855774, -1.715022087097168, -1.7033874988555908, 0.13489215075969696, -1.9168601036071777, 0.2731603682041168, -1.189683437347412, -0.18437166512012482, 2.0865166187286377, 0.8831823468208313, 0.5650487542152405, -0.4825064241886139, -1.478804588317871, -2.746879816055298, 1.673429250717163, -0.5879411101341248, 0.7389925122261047, 0.38353389501571655, -2.608180522918701, 0.027355443686246872, -2.125446081161499, 1.6552927494049072, -0.4944746196269989, -3.110832929611206, -2.390852212905884, -0.03128872066736221, -0.24275508522987366, 2.335111618041992, 1.1099071502685547, 1.7365093231201172, -0.7652358412742615, -1.4650542736053467, 0.9305098056793213, -0.3189709186553955, 1.4164842367172241, -1.5589946508407593, 0.3885362148284912, 2.055479049682617, -1.8997472524642944, -2.21557879447937, 0.302901029586792, 1.7605160474777222, -1.9056901931762695, 1.610260248184204, 3.423177480697632, -0.3988611102104187, 0.06269221752882004, 2.5522842407226562, 0.7584719657897949, -0.7215784192085266, -1.5704973936080933, 3.7611007690429688, 0.09241728484630585, -0.5561078786849976, 2.4581403732299805, -1.6346186399459839, 0.27076050639152527, 0.9742812514305115, -2.547398090362549, 0.45163998007774353, 1.3837000131607056, 0.8487511873245239, -0.05777039751410484, -0.04063934460282326, 1.2839199304580688, 0.4774395227432251, -1.0750194787979126, 0.07563666254281998, -1.1563507318496704, 0.9532269239425659, -2.542055368423462, 2.059359550476074, 1.0191620588302612, 2.040602684020996, -2.5682530403137207, -0.5640752911567688, 0.6122928857803345, -2.0178306102752686, 0.9562038779258728, 0.3582436740398407, -0.769043505191803, 1.4265239238739014, 0.22405341267585754, -1.2743690013885498, 0.3596164286136627, -0.7374376654624939, -0.0064550829119980335, -1.0919266939163208, 0.42650628089904785, 2.254056930541992, 1.0435810089111328, 1.73190176486969, 0.6610095500946045, 0.3010731339454651, -3.3484785556793213, -1.0809575319290161, -0.9526941180229187, -2.073613166809082, 0.9570202827453613, 1.969017744064331, 1.095607042312622, 0.4928782880306244, 2.7001352310180664, -1.367802619934082, -0.10481774806976318, 0.4562912881374359, -1.656671404838562], "sim": 0.413044273853302, "freq": 353.0, "x_tsne": 16.69577407836914, "y_tsne": 17.919036865234375}, {"lex": "prophylactics", "type": "nb", "subreddit": "Coronavirus", "vec": [0.2459053099155426, -0.26304036378860474, 0.7934380769729614, -0.4278259873390198, -0.3814258277416229, 0.22543609142303467, -0.008030966855585575, -0.2002871036529541, 0.2974948585033417, -0.24845914542675018, -0.06377214193344116, 0.38411691784858704, 0.8198341727256775, -0.3073141276836395, 0.7975727319717407, 0.10008689016103745, 0.058601461350917816, -0.09725695848464966, 0.9528362154960632, 0.5789309740066528, -0.09118441492319107, 0.032142139971256256, 0.03202427178621292, 0.5658054351806641, 0.7385250926017761, 0.15451490879058838, 0.3345070779323578, 0.1824418008327484, -0.05252665653824806, -0.007526264060288668, 0.2765275835990906, 0.49728041887283325, 0.44909459352493286, -0.2673875093460083, 0.3012121915817261, 0.406596302986145, -0.8225950002670288, 0.29402995109558105, -0.1254965215921402, -0.1443343460559845, -0.34061774611473083, -0.01153936143964529, -0.30722174048423767, 0.702453076839447, 0.8657249212265015, 0.023946380242705345, 0.37772566080093384, 0.7114723324775696, -0.0639367550611496, 0.29618725180625916, -0.17703688144683838, -0.5610510110855103, -0.14306750893592834, 0.026836389675736427, -0.9047630429267883, 0.5293257832527161, 0.1712760329246521, 0.02484242431819439, -0.14888478815555573, -0.9312039017677307, -0.09725016355514526, -0.16632595658302307, -0.41142910718917847, 0.281884104013443, 0.6616692543029785, 0.6212654113769531, -0.18015548586845398, 0.5408363938331604, 0.08763722330331802, 0.29957374930381775, -0.08028055727481842, -0.7690026164054871, -0.026891835033893585, -0.24139414727687836, -0.3016475737094879, -0.1677868515253067, -0.036079928278923035, 0.34198182821273804, 0.05843762680888176, 0.058424752205610275, -0.04513298720121384, -0.055160511285066605, -0.26052120327949524, 0.010172931477427483, -0.23875276744365692, 0.128622367978096, 0.20351171493530273, 0.2282184213399887, -0.16496628522872925, -0.05444850027561188, -0.2323053926229477, -0.14228829741477966, -0.08780846744775772, 0.266165167093277, -1.0006904602050781, -1.3031898736953735, 0.059879157692193985, -0.4325326085090637, 0.23220455646514893, 0.2811287045478821, -0.5071427822113037, -0.1985766738653183, 0.32661688327789307, -0.3250085413455963, -0.49246907234191895, -0.4100196361541748, 0.5173559784889221, -0.30825522541999817, 0.47664013504981995, -0.1589224934577942, -0.0007011130801402032, 0.18573980033397675, 0.6073618531227112, -0.14356043934822083, -0.5961204171180725, -0.2958258092403412, -0.06302805989980698, 0.3040291368961334, 0.5462898015975952, -0.2701377868652344, -0.15026314556598663, 0.2557392418384552, -0.1275627315044403, 0.3954773545265198, 0.19603319466114044, 0.9080527424812317, -0.26656559109687805, -0.25471267104148865, 0.4354092478752136, 0.3778727650642395, 0.17067892849445343, -0.08107957243919373, -0.5015424489974976, 0.10830739885568619, -0.5231407880783081, -0.5741298794746399, -0.46910759806632996, -0.41083040833473206, -0.5516555905342102, 0.028941968455910683, 0.26163527369499207, 0.12822981178760529, 0.24300718307495117, 0.20880255103111267, 0.041523657739162445, -0.2441556602716446, -0.1200515404343605, -0.41634973883628845, 0.2899850606918335, 0.0020406560506671667, 0.05427535995841026, -0.046917617321014404, 0.3743155300617218, 0.172920361161232, 0.014494716189801693, -0.17413203418254852, -0.24876664578914642, 0.29190823435783386, -0.6212847828865051, 0.2555616796016693, 0.12976832687854767, -0.13112370669841766, -0.01162161398679018, 0.1440601795911789, -0.1512090116739273, 0.48573508858680725, -0.639582097530365, 0.594362735748291, 0.05982603132724762, -0.09359763562679291, -0.5747068524360657, 0.056184977293014526, 0.5104697942733765, 0.3334498405456543, 0.056724462658166885, -0.14925120770931244, 0.36616265773773193, 0.3410877287387848, -0.6111257672309875, -0.1795801818370819, 0.373261421918869, 0.4726051688194275, -0.09546684473752975, 0.47359541058540344, -0.07525660842657089, 0.07101158052682877, 0.8030912280082703, -0.24660764634609222, -0.14250732958316803, -0.2227170765399933, -0.029912525787949562, -0.23193958401679993, 0.026211902499198914, 0.19039879739284515, 0.09488128870725632, -0.09680070728063583, -0.030118068680167198, 0.1815362423658371, -0.014166361652314663, 0.19679571688175201, -0.476995587348938, 0.5428336262702942, -0.469015508890152, 0.5895716547966003, -0.5076156258583069, 0.35878753662109375, -0.09953399002552032, -0.2958310544490814, 0.43507200479507446, 0.46549615263938904, 0.13658905029296875, 0.26897957921028137, -0.4966164827346802, -0.2404577136039734, 0.26096561551094055, 0.5502203106880188, -0.4718388617038727, -0.4105973541736603, -0.5935674905776978, -0.8797227144241333, 0.3484916388988495, 0.2829475402832031, -0.30435922741889954, 0.29989635944366455, 0.40377676486968994, -0.39245572686195374, -0.14441408216953278, 0.24936926364898682, -0.38440269231796265, 0.08375218510627747, 0.4405284523963928, 0.45552724599838257, 0.27871984243392944, 0.4661748707294464, 0.2268180549144745, 0.32937270402908325, 0.24586111307144165, 0.06582638621330261, -0.09433971345424652, -0.41301754117012024, -0.5966173410415649, 0.3337424397468567, -0.20584258437156677, 0.10577838867902756, -0.09734918177127838, -0.16800270974636078, 0.06607406586408615, 0.3084522485733032, 0.3767785131931305, 0.7367436289787292, 0.619906485080719, -0.447644978761673, 0.5512294769287109, 0.0676339790225029, 0.35062333941459656, -0.304036021232605, -0.25974375009536743, 0.12968163192272186, 0.29081419110298157, -0.7179265022277832, -0.10991951823234558, -0.37933534383773804, -0.7960633039474487, -0.289873868227005, -0.14025554060935974, -0.533176839351654, 0.1347874104976654, 0.2847962975502014, 0.5622804760932922, -0.3833349645137787, 0.28214195370674133, -0.5552539825439453, -0.1791137158870697, 0.386972576379776, 0.5286456346511841, 0.8257574439048767, -0.24228020012378693, 0.6532354354858398, 0.026660297065973282, -0.35589849948883057, -0.40433719754219055, -0.33858489990234375, -0.5067196488380432, 0.5049620270729065, 0.5058647394180298, 0.766074538230896, -0.44789919257164, 0.20079439878463745, 0.028861045837402344, -0.0254548080265522, 0.5038385987281799, -0.3238348066806793, -0.03088897466659546, 0.09677151590585709, 0.8780821561813354, 0.12254423648118973, -0.25722891092300415, -0.313633531332016, -0.11015741527080536, -0.3211970925331116], "sim": 0.4123225808143616, "freq": 20.0, "x_tsne": -4.490015983581543, "y_tsne": 22.916650772094727}, {"lex": "drugmakers", "type": "nb", "subreddit": "Coronavirus", "vec": [0.15458644926548004, 0.8455367088317871, 0.3977477550506592, -0.5053501129150391, -0.441222220659256, 1.311401605606079, 0.16271857917308807, -0.25670790672302246, -0.5120255947113037, -0.7255929112434387, -0.11903715133666992, 1.1260780096054077, 0.9536150693893433, -0.897820770740509, 0.13306617736816406, 0.40982353687286377, -0.5376403331756592, 0.35683271288871765, 0.23195651173591614, -0.37806007266044617, -0.49389612674713135, -0.3525339663028717, -1.2747416496276855, 0.01651867665350437, 0.11763371527194977, -0.2869865894317627, 0.2427596151828766, 0.8724060654640198, 0.598879873752594, -0.2447342872619629, -0.037497617304325104, -0.8182005882263184, 0.489062637090683, 0.3945176303386688, 0.8887770771980286, -0.05751051753759384, -0.7084073424339294, 0.05707933008670807, -1.1151633262634277, 0.4085697829723358, -1.076309084892273, 1.5402361154556274, 0.24099810421466827, 0.21322816610336304, 0.22838741540908813, -0.35022345185279846, 0.30198967456817627, 0.21721595525741577, -0.28357985615730286, -0.8199096322059631, -0.1547529399394989, 1.2256578207015991, -0.2873990535736084, -0.23559756577014923, 0.7146247029304504, 0.5259667038917542, 0.46994897723197937, -1.0595368146896362, -0.3311821222305298, -1.139811396598816, 0.28313344717025757, 0.12235046178102493, 0.07819727808237076, -0.7883398532867432, -0.2727991044521332, -0.25442126393318176, -0.786066472530365, -1.0160409212112427, -0.2697816789150238, 0.18311691284179688, -0.15732567012310028, -0.1034659892320633, -1.3230023384094238, -0.8169927597045898, -0.2199522852897644, -1.1212108135223389, 1.0470620393753052, 0.4755587577819824, 0.41120025515556335, 0.3473542332649231, 0.5399867296218872, 0.5213792324066162, -0.7636024355888367, -0.19116182625293732, -0.06947755068540573, 0.357769638299942, 0.1445792317390442, -0.789251983165741, -0.3640346825122833, -0.02677333354949951, -0.03173171728849411, -0.33215489983558655, 0.3101159334182739, 0.34658217430114746, -0.21227511763572693, -0.8404293656349182, 0.39220643043518066, -0.5463767647743225, 0.02446986921131611, 0.25699278712272644, 0.34101390838623047, -0.09423400461673737, 1.2142088413238525, -0.662166178226471, -0.34854212403297424, -0.19336248934268951, 0.7113364338874817, -0.09322274476289749, 0.2847374379634857, 0.24821558594703674, -0.5938643217086792, 0.4724523723125458, -0.3066437542438507, -0.890118420124054, 0.33402034640312195, -0.3156903386116028, 0.017032340168952942, -0.6686737537384033, -0.008183938451111317, 0.21695272624492645, -0.736169695854187, -0.5467754006385803, -0.2451997548341751, -1.0000231266021729, -0.9870901703834534, 1.6991034746170044, -0.9800766706466675, 0.06628600507974625, 0.15847550332546234, -0.8828681111335754, -0.16466207802295685, -0.6068935394287109, -1.197975993156433, -0.017624452710151672, 0.02571374736726284, -0.2616890072822571, -0.2843094766139984, 0.37860366702079773, 0.5961214303970337, -0.6559106707572937, 0.2522681951522827, -0.7916745543479919, 1.0933400392532349, 0.49746620655059814, -0.08932939171791077, -0.14920929074287415, -0.9590590000152588, -0.7007423043251038, 0.1859925389289856, -1.3922544717788696, 0.9914746880531311, 1.3586890697479248, 0.5136191248893738, 0.6318091750144958, -0.021404100582003593, 0.015306876972317696, -0.35783058404922485, -0.3880060613155365, -0.003012196160852909, 0.048579633235931396, 0.24431729316711426, 0.5456889271736145, 0.005694179330021143, -0.16196320950984955, 0.01334096398204565, 0.27135998010635376, 0.09719233959913254, 0.20704585313796997, 0.19882823526859283, 0.6128733158111572, -0.9166346192359924, -0.687322199344635, -0.1710847020149231, 1.0940492153167725, -0.4388403594493866, -0.6314998269081116, -0.26956048607826233, -0.9748583436012268, 0.05804459750652313, -0.6522802114486694, 0.8173863887786865, -0.12734974920749664, 0.8414702415466309, 0.07014235854148865, -0.7359310388565063, 0.013789570890367031, 0.7681017518043518, 0.5918513536453247, -0.6023374199867249, -0.3850102424621582, -0.10143988579511642, 0.0963159054517746, -0.7334333062171936, -0.243687242269516, 0.004273320082575083, -0.44531574845314026, 0.24196462333202362, -0.048455338925123215, 0.5167523622512817, 0.27489572763442993, -0.4909238815307617, 0.6032170057296753, -0.1858592927455902, 0.3507997989654541, -0.5628683567047119, -0.66778165102005, -1.3973352909088135, 1.1012234687805176, -0.1141211986541748, -0.41762739419937134, 0.4696787893772125, 0.06220848858356476, -0.9215719699859619, 0.6873044371604919, 0.43490737676620483, -0.3409543037414551, -0.12427239865064621, 0.08337422460317612, 0.8920362591743469, -0.4159448444843292, 0.6716688275337219, 0.6398075222969055, -0.7889399528503418, 0.1395508348941803, -0.3409577012062073, -0.12229660898447037, 0.2404976785182953, 0.21263983845710754, 0.26908716559410095, -0.58469158411026, -0.011111407540738583, -0.17734386026859283, -0.16288509964942932, -0.08015386760234833, 1.4358978271484375, 0.1553276777267456, -0.32772746682167053, 1.3357481956481934, -0.7753530144691467, -0.17301376163959503, -0.16391156613826752, -0.06314096599817276, -0.25255146622657776, -0.8195275664329529, 0.27109479904174805, 0.054459817707538605, 0.2274833768606186, 0.12326756119728088, 0.07579527050256729, 0.8540052771568298, 0.6113393902778625, -0.8283914923667908, 0.3798442780971527, 0.8092628717422485, -0.5852031111717224, 0.6604742407798767, 0.8818275928497314, -0.14530901610851288, 0.32216307520866394, -1.2793101072311401, -0.28287848830223083, -0.31871655583381653, -0.6836019158363342, -0.6515419483184814, 0.6618334650993347, -0.25763437151908875, 0.3181077241897583, 0.27663668990135193, 0.6467147469520569, 0.2129780650138855, 0.2530399262905121, 0.359138548374176, 0.12147761136293411, -0.061718329787254333, 0.4335273206233978, 0.7295054197311401, 0.17055118083953857, 0.3109130263328552, 0.042170681059360504, -0.7279689311981201, 0.05383174121379852, -0.5141751170158386, -0.36816704273223877, 0.5848774909973145, -0.670582115650177, -0.2536008656024933, -0.408557653427124, -0.522767186164856, 0.23147521913051605, -0.3123769760131836, -0.3201325237751007, 0.4023882746696472, 0.6664525866508484, -0.7788048982620239, -0.24047239124774933, -0.41930145025253296, -0.36938828229904175, 0.2938833236694336, -0.506325364112854, 0.8362599015235901], "sim": 0.4055165648460388, "freq": 37.0, "x_tsne": 19.303180694580078, "y_tsne": 0.8445531725883484}, {"lex": "precursors", "type": "nb", "subreddit": "Coronavirus", "vec": [0.21907176077365875, -0.4825976490974426, -0.2781495153903961, -1.3948814868927002, -1.1273177862167358, 0.652108371257782, 0.5282710194587708, -0.272878497838974, 0.3375234603881836, -0.027361085638403893, -0.14230626821517944, 0.06401430070400238, 0.8076444864273071, -0.9090872406959534, 0.30507364869117737, -0.2534633278846741, -0.11267640441656113, -0.48322755098342896, -0.06499917060136795, -0.8269848823547363, -1.1602044105529785, 0.014368404634296894, -0.778098464012146, 0.42973092198371887, 0.0176392812281847, 0.0741209164261818, -0.07211976498365402, 0.9678639769554138, -0.4351103603839874, 0.11492880433797836, 1.517444372177124, 0.29466497898101807, 0.2908634841442108, -0.4113899767398834, 0.5736663341522217, 0.11927465349435806, -1.0364990234375, -1.0155514478683472, -0.24509556591510773, 0.26097390055656433, -1.0753434896469116, 0.7483641505241394, 0.716686487197876, 0.9942978620529175, -0.46612995862960815, -0.9320642948150635, 1.2423677444458008, 1.261968731880188, -0.689725399017334, -0.13095910847187042, 0.09231655299663544, 0.29031383991241455, -0.5823541879653931, 0.581473708152771, -1.0878678560256958, 0.07157764583826065, -0.20362354815006256, 0.32199886441230774, -0.3932092785835266, -0.4517231583595276, 0.1670929342508316, 0.1906537115573883, 0.4545694887638092, 0.28402963280677795, 0.145409494638443, 0.10031113773584366, -0.026526419445872307, -0.462558388710022, 0.8467943668365479, 0.9239194989204407, 0.2143789678812027, 0.04637699946761131, 0.5025390386581421, -0.9650377631187439, -0.5793850421905518, -0.667188823223114, -0.02614622376859188, 0.3422475755214691, 0.21548181772232056, 0.8093495965003967, 0.3401404917240143, -0.026310045272111893, -0.5075218081474304, -1.057066798210144, 0.15567217767238617, 0.9657576084136963, 0.904982328414917, 0.9589641094207764, 0.45973441004753113, 0.5850449800491333, -0.36824944615364075, -1.128827691078186, 0.8499560356140137, 0.15065599977970123, -0.6019479632377625, 0.07264325767755508, 0.6070660352706909, -0.17618988454341888, 0.49523288011550903, 0.2415819764137268, 1.0774527788162231, 0.12376111000776291, 0.7556383609771729, -1.0757309198379517, -0.7865743637084961, 0.056413684040308, -0.27284348011016846, -0.03449326381087303, -0.5634106993675232, -0.5873662233352661, -1.020544171333313, -1.1951448917388916, 0.31115567684173584, -0.22652667760849, -0.41747045516967773, -0.23219341039657593, -0.2665298283100128, -0.235334113240242, -0.35308554768562317, 0.5568992495536804, 0.7373600006103516, -0.4249274432659149, 0.7136325836181641, -0.08067587018013, -0.3368782103061676, 1.3262239694595337, -0.746573269367218, -0.5563613176345825, 0.6737174391746521, -1.149680256843567, -0.009801192209124565, 0.06442617624998093, -0.6729059815406799, 0.5303313136100769, 0.6557819843292236, 0.07306346297264099, -1.389207363128662, -0.21552768349647522, 0.27270328998565674, -0.6068898439407349, -0.10523686558008194, 0.5422768592834473, 0.33312422037124634, 0.22479616105556488, -0.24394114315509796, -0.4341469407081604, -1.2231671810150146, 0.06649862229824066, 0.3845943212509155, 0.014243529178202152, -0.2621316611766815, 0.11936575919389725, 0.4356718957424164, 0.043118566274642944, 0.6079235076904297, -0.3527431786060333, -0.6583379507064819, -0.5602920055389404, 0.5351542830467224, -0.32815006375312805, 0.6862120032310486, -0.12745735049247742, -0.3922950029373169, 0.47223350405693054, -0.28650185465812683, 0.27324140071868896, 0.3472612798213959, 1.092003345489502, -0.8471844792366028, 0.787932276725769, -0.054968591779470444, -0.04811146855354309, -1.2408109903335571, 0.5047397017478943, -0.6430683732032776, 0.0025663417764008045, -0.8728989958763123, -0.3946610689163208, 0.6235833764076233, -0.5758454203605652, 0.38556671142578125, 1.1505467891693115, 0.26043298840522766, 0.664170503616333, -0.5977256298065186, 0.4232363700866699, 1.5530314445495605, 0.20591120421886444, 0.687322735786438, -0.027609169483184814, 0.17924994230270386, -0.8584951758384705, 0.043472472578287125, 0.009083704091608524, -0.16860634088516235, -1.1197818517684937, 0.11149660497903824, 0.08211827278137207, 1.10984206199646, 0.184051513671875, -1.0206421613693237, -0.4469563663005829, 0.3490064740180969, 1.3442316055297852, -0.43140122294425964, 0.09013400226831436, -0.7423038482666016, -0.17229239642620087, 0.3573117256164551, 0.6882093548774719, -0.36772313714027405, -0.5876864790916443, -0.9676206707954407, 0.6027803421020508, -0.03850661590695381, 0.16621923446655273, -0.7084019780158997, -1.1253726482391357, -0.08392284065485, -0.5036597847938538, 0.4653015732765198, 1.2481191158294678, -0.3276797831058502, -0.18469442427158356, 0.22516533732414246, 0.09836533665657043, 0.36691969633102417, -0.3532814681529999, -0.26941198110580444, 0.5870433449745178, 1.0108134746551514, 0.16299648582935333, -0.27925845980644226, 0.5096709132194519, 1.2954758405685425, -1.2578188180923462, 0.5333666205406189, 0.8913041353225708, -0.45342275500297546, 0.3008977770805359, -0.08051562309265137, 0.11430758982896805, -0.9481306076049805, 0.32012107968330383, 0.2945595979690552, 0.5589582324028015, 0.7557744979858398, 0.8132714033126831, 0.868032693862915, 0.25143226981163025, -0.3318970501422882, -0.8248390555381775, 0.9639482498168945, -0.6619239449501038, -0.5195484161376953, -0.8693552017211914, -0.9372857213020325, 0.29953533411026, -0.03664550185203552, 0.29981333017349243, 0.1728476881980896, -0.6102758049964905, -0.8227211236953735, -1.366028904914856, -0.2591951787471771, 0.07872240245342255, 0.3534490168094635, -0.24905596673488617, 0.6018944382667542, 0.34259435534477234, -0.8047856092453003, 0.06220700591802597, 0.20250928401947021, 0.05658897012472153, 1.2114839553833008, 0.5933680534362793, -0.8896281719207764, 0.7607274651527405, 0.09510885924100876, -0.09593596309423447, -0.7785306572914124, 0.36722269654273987, -0.8361340165138245, 0.7358202338218689, 0.13818009197711945, 0.09437685459852219, -0.09822209924459457, -0.7004681825637817, 0.33791089057922363, 0.3644050061702728, -0.5602904558181763, 0.295728862285614, 0.5415835380554199, 0.07992693781852722, -0.11057385802268982, 0.22859394550323486, 0.46168583631515503, 0.3411528170108795, 0.1656886786222458, 0.0857846587896347], "sim": 0.4020351469516754, "freq": 59.0, "x_tsne": 20.31438446044922, "y_tsne": 18.12069320678711}, {"lex": "preventions", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.2028028666973114, -0.37756556272506714, 0.1114499568939209, -0.3108989894390106, -0.3753722012042999, 0.43431079387664795, 0.22799083590507507, -0.6994791626930237, -0.4241366386413574, 0.17339755594730377, 0.5657933354377747, 0.2679096758365631, 0.12887664139270782, -0.14983920753002167, 0.2997969090938568, 0.25255632400512695, 0.3130141496658325, -0.05667375028133392, 0.32111242413520813, 0.2189825475215912, 0.08154811710119247, 0.49005553126335144, 0.18433614075183868, 0.4886131286621094, 0.718822717666626, -0.16226841509342194, 0.28797635436058044, 0.16225257515907288, -0.028303708881139755, -0.31075045466423035, 0.44852229952812195, -0.04453255608677864, 0.03456806391477585, -0.7156765460968018, 0.2024558186531067, -0.03350962698459625, -0.2669561803340912, 0.35107070207595825, -0.06366000324487686, -0.09805939346551895, -0.10177339613437653, -0.039294272661209106, -0.06974007189273834, 0.294528603553772, 0.7363048791885376, -0.10520114749670029, 0.5669757127761841, 0.4789072275161743, 0.27615392208099365, 0.38368186354637146, -0.20282603800296783, 0.19336797297000885, -0.2609732151031494, -0.4243741035461426, -0.48595187067985535, -0.04730784893035889, 0.30510735511779785, 0.47650089859962463, 0.10437215864658356, -0.1521349847316742, -0.2667844593524933, 0.5652766823768616, -0.22132305800914764, 0.015672674402594566, -0.13696332275867462, 0.8197727799415588, -0.26384904980659485, -0.33549532294273376, 0.008733372204005718, -0.03237879276275635, 0.17014926671981812, -0.2611960470676422, 0.06840953975915909, 0.48057618737220764, 0.14815080165863037, 0.138130322098732, 0.0677734911441803, 0.34298205375671387, -0.2974540591239929, 0.17272572219371796, 0.31512773036956787, 0.26457324624061584, 0.12862925231456757, 0.06817866116762161, 0.08478989452123642, 0.05843457952141762, -0.4554630219936371, 0.29305100440979004, 0.1810949146747589, 0.1271354854106903, -0.27588507533073425, -0.45577171444892883, 0.19799496233463287, 0.1863522082567215, -0.9826189875602722, -0.3490608036518097, -0.3178781270980835, -0.4058588743209839, -0.039386000484228134, -0.36301782727241516, -0.02013234980404377, 0.2847897410392761, -0.2938518822193146, 0.23949845135211945, -0.17215010523796082, -0.43199798464775085, 0.7008214592933655, -0.06658809632062912, 0.2588210701942444, -0.3981688916683197, 0.39212939143180847, -0.16621045768260956, -0.11736045032739639, -0.18970949947834015, -0.161840558052063, -0.9475463628768921, -0.29193007946014404, -0.34119608998298645, 0.09011884033679962, -0.1127278208732605, -0.11968768388032913, 0.01581421308219433, -0.28269368410110474, -0.12931503355503082, -0.17565672099590302, 0.6007506847381592, 0.09412439167499542, 0.01401693094521761, 0.4308089017868042, -0.22592273354530334, 0.028217753395438194, 0.4545624852180481, -0.3086969256401062, -0.35831791162490845, -0.009631220251321793, -0.1694527268409729, 0.14713509380817413, 0.13054436445236206, 0.19890759885311127, 0.3004039227962494, 0.43285655975341797, -0.08871935307979584, 0.0651787519454956, -0.39724433422088623, 0.3271218538284302, -0.07357332855463028, 0.3830505311489105, -0.23781034350395203, -0.1975342333316803, -0.3980286419391632, 0.22678425908088684, -0.38331368565559387, 0.7667103409767151, -0.27337056398391724, -0.1011144369840622, 0.16211262345314026, -0.21208880841732025, 0.00489066494628787, -0.4188462197780609, 0.16881975531578064, 0.07060723751783371, 0.08362461626529694, 0.3502369225025177, 0.34814453125, 0.16061235964298248, -0.07223879545927048, 0.1536117047071457, 0.9216159582138062, 0.017911437898874283, 0.02233114093542099, -0.3408765196800232, -0.2241218239068985, 0.22920522093772888, 0.0522526279091835, 0.4074687659740448, -0.15000198781490326, -0.17032794654369354, 0.3045911192893982, -0.016345173120498657, -0.010390964336693287, -0.46255022287368774, 0.7394896745681763, 0.4277594983577728, -0.2781710922718048, -0.09960170835256577, 0.2651808559894562, 0.31207355856895447, -0.38725849986076355, 0.11503318697214127, 0.2584722340106964, -0.012952608987689018, -0.38846877217292786, 0.38554516434669495, 0.32527831196784973, -0.29680442810058594, 0.3230110704898834, -0.6414258480072021, -0.32292184233665466, -0.2890533208847046, 0.4961390495300293, -0.3717527389526367, 0.2658151090145111, -0.128627747297287, 0.12019931524991989, -0.8058250546455383, -0.01780170574784279, 0.010672804899513721, 0.15209050476551056, 0.17653237283229828, 0.2488054782152176, 0.032906435430049896, -0.13569022715091705, -0.12189725786447525, -0.09064804762601852, -0.0451958067715168, 0.13338136672973633, 0.372196763753891, -0.15903082489967346, 0.3664458096027374, -0.5049142241477966, -0.04415754973888397, -0.002872662153095007, 0.11211054027080536, -0.209494486451149, 0.3145546615123749, -0.4427490532398224, 0.22631599009037018, 0.11255326867103577, -0.09960837662220001, -0.5520122051239014, 0.054223574697971344, -0.12390650063753128, 0.6684592962265015, -0.1746836155653, 0.4834269881248474, -0.12117241322994232, -0.12355796992778778, 0.24848805367946625, 0.1797100156545639, -0.08293876051902771, -0.07056467980146408, 0.12070439010858536, -0.5556631088256836, -0.1701221913099289, 0.18797330558300018, 0.04535331949591637, 0.4638639986515045, 0.6004330515861511, 0.2487613707780838, 0.6824302077293396, -0.10236422717571259, 0.034838784486055374, 0.09076564013957977, 0.19610166549682617, 0.05638004094362259, 0.1134335994720459, 0.005607070866972208, 0.253686785697937, -0.5304408073425293, -0.3155106008052826, -0.3850201666355133, 0.05132823437452316, -0.29749444127082825, 0.47578465938568115, 0.45275136828422546, -0.12114932388067245, -0.07608148455619812, -0.001941588707268238, 0.03985931724309921, -0.041318874806165695, 0.048018742352724075, -0.28806135058403015, -0.5647173523902893, 0.07682649046182632, 0.4402787387371063, -0.0649496540427208, -0.5582590103149414, 0.4861380159854889, 0.1620371788740158, -0.32080602645874023, 0.07219567894935608, -0.014330729842185974, -0.10941611975431442, 0.35532328486442566, -0.1180359274148941, 0.3838997781276703, -0.49228665232658386, -0.06626760959625244, -0.6932917833328247, 0.44630104303359985, 0.2377137541770935, -0.23859472572803497, -0.14302648603916168, -0.45581454038619995, 0.17543815076351166, -0.627559244632721, 0.3004521131515503, -0.5258211493492126, -0.03769554942846298, -0.22619393467903137], "sim": 0.40006163716316223, "freq": 20.0, "x_tsne": 17.352983474731445, "y_tsne": -15.624991416931152}, {"lex": "versions", "type": "nb", "subreddit": "Coronavirus", "vec": [0.6945297122001648, -0.6458688378334045, 0.970847487449646, -1.2517169713974, 1.1156498193740845, -1.202329397201538, 1.7283703088760376, -1.8240970373153687, -0.5287006497383118, -0.48025521636009216, 0.40809839963912964, -0.6915164589881897, 3.0129740238189697, -1.5789166688919067, -0.32009848952293396, -1.4072916507720947, -0.6720407605171204, -0.7566486597061157, 0.681845486164093, -0.6346361637115479, -0.3414762318134308, -0.23115749657154083, -1.1058679819107056, -0.6885400414466858, 3.9667019844055176, 0.23722530901432037, -0.8467952013015747, -2.470898151397705, -0.25632184743881226, -1.8479012250900269, -3.52333402633667, -0.7299486398696899, -1.8046784400939941, -1.0336406230926514, 0.9909518361091614, 2.3850655555725098, -0.3115110695362091, -0.38134199380874634, -3.6932365894317627, -1.0842353105545044, -0.15909314155578613, 2.566793918609619, 0.4827113747596741, -1.930622935295105, 0.24412812292575836, 1.2674161195755005, 0.09051408618688583, 1.2823408842086792, -1.2620460987091064, -1.4475103616714478, -1.5336586236953735, 1.1219482421875, -0.530125617980957, 0.881546139717102, -0.041454609483480453, 2.405062198638916, -0.28899234533309937, -1.3060302734375, 0.17972432076931, -2.5686123371124268, 1.2648462057113647, -0.10165604948997498, 1.5552864074707031, -1.847640872001648, 0.6734301447868347, 1.4247033596038818, 1.7619283199310303, -1.759928584098816, -3.3871426582336426, 0.18960809707641602, 2.5609779357910156, -1.0307371616363525, 2.2791495323181152, 0.5695172548294067, -0.958998441696167, -1.37405526638031, 1.8022689819335938, 1.973970651626587, -0.8091389536857605, 2.387789011001587, 2.7208316326141357, -0.028627809137105942, -2.3237736225128174, -0.032063134014606476, -3.2448716163635254, -1.5200945138931274, -1.7732716798782349, 0.5684417486190796, -1.0868487358093262, 2.019831418991089, -1.159596562385559, -0.009508232586085796, 0.22913140058517456, -0.5264211297035217, 0.04655000939965248, -0.7075026035308838, -1.3766311407089233, -0.6766443252563477, 1.60052490234375, -0.2406119853258133, -0.6344555020332336, 0.8270775079727173, 0.9706989526748657, -0.6522893309593201, -2.1485097408294678, 0.8240640759468079, 0.3437977433204651, -0.7651370167732239, 2.3992116451263428, -1.473562240600586, 1.4578793048858643, -2.0324673652648926, -0.4241941571235657, -2.9097609519958496, -0.5890548825263977, -0.13547614216804504, -0.21106010675430298, -0.9797812700271606, 1.645148754119873, 1.5382046699523926, 0.23919840157032013, -1.0613609552383423, -2.573723077774048, -2.9183266162872314, 0.39944836497306824, 2.051668405532837, -0.5988393425941467, 2.9161126613616943, 0.05146702006459236, 0.8086885213851929, 0.7985966205596924, -3.480759859085083, -2.4580039978027344, 0.07403740286827087, -0.6786469221115112, 0.3206583261489868, -1.2391983270645142, -0.2691113352775574, -0.21309538185596466, 0.44186586141586304, 0.7239879369735718, -1.2759836912155151, 0.7330996990203857, -0.8993143439292908, 0.0930461436510086, 0.9402850270271301, 0.5364169478416443, -2.816030979156494, 1.5078930854797363, 0.5078018307685852, 1.319574236869812, -1.2532134056091309, 0.8796995878219604, 0.7164347171783447, 0.0345965139567852, 0.7056818008422852, 4.671667575836182, -1.509892463684082, -1.2987487316131592, -0.7111969590187073, -1.637924313545227, 1.248358964920044, -0.3236134350299835, 2.539045572280884, -0.9644896984100342, 0.6039336323738098, -0.848160445690155, -0.6805177330970764, 0.553550660610199, 1.0387681722640991, -0.5659658908843994, 0.8138579726219177, -0.15307271480560303, -1.448090672492981, -0.08114209771156311, -0.4117580056190491, -0.6790114641189575, -1.2546933889389038, 3.0492944717407227, -1.7581589221954346, -1.9055685997009277, 0.3799650967121124, 0.9833692908287048, 0.06305909901857376, -1.5835657119750977, -1.1330606937408447, 2.6619174480438232, -1.1078295707702637, 0.4353847801685333, -1.573058009147644, -0.5332739949226379, -0.05276641249656677, -0.9560979008674622, 0.915046215057373, -1.068721890449524, -1.8351531028747559, 0.8297964930534363, -0.38979947566986084, 1.0589946508407593, 0.7031344175338745, -3.3924081325531006, -0.5003177523612976, -0.531596839427948, 0.7777553796768188, -0.748691201210022, -1.6881821155548096, -2.3740150928497314, 2.8604931831359863, 2.3589420318603516, 0.6116913557052612, -1.251057744026184, 0.9023066163063049, -0.6201410889625549, 0.1366928666830063, 1.3291643857955933, -0.0014691537944599986, -1.3598638772964478, 0.2664682865142822, -1.2596209049224854, 0.3365938067436218, 1.3084348440170288, 1.8873584270477295, 0.8311078548431396, 0.3152385652065277, -0.1435210406780243, -0.32713180780410767, 1.1499743461608887, 0.7193617224693298, -1.3075416088104248, -1.3168941736221313, -0.17445281147956848, -2.4072487354278564, -1.3718665838241577, -0.06804542243480682, 1.4049551486968994, -2.247718095779419, -0.3005965054035187, 1.2210036516189575, -0.5334234237670898, 0.5109751224517822, -0.8693445324897766, -1.0885597467422485, -0.3387409746646881, -2.7675483226776123, -1.7490034103393555, -0.7614959478378296, 1.5181723833084106, 0.7523310780525208, -0.2879799008369446, 0.0591072216629982, -2.1089258193969727, 0.47003889083862305, 0.15721091628074646, 0.30462366342544556, -1.1528583765029907, 0.09276919066905975, -0.8709007501602173, -1.444782018661499, 0.391035258769989, -2.3444089889526367, 0.612373411655426, 0.9362385272979736, -1.6674070358276367, -1.413865327835083, 1.1501792669296265, 0.8227735161781311, 0.7106266617774963, 1.8762329816818237, -0.7036459445953369, 2.5177817344665527, -0.022034287452697754, 1.3435925245285034, -0.6403160691261292, 0.6519013047218323, 2.3647074699401855, 0.7062699198722839, -0.049733541905879974, 3.270181655883789, 0.5076395273208618, -0.2293456643819809, -0.28494709730148315, 0.4358620345592499, -2.23898983001709, 2.570399045944214, 0.999106764793396, -1.8399988412857056, -1.885315179824829, -1.1935878992080688, -1.5759615898132324, 1.093603491783142, -1.764105200767517, -0.5540117621421814, 1.5720593929290771, -1.1972424983978271, 1.776923656463623, -0.2555929720401764, -1.9212372303009033, -0.3010542094707489, -2.526719570159912, -2.3237435817718506], "sim": 0.39596495032310486, "freq": 1013.0, "x_tsne": 2.980243682861328, "y_tsne": 20.491540908813477}, {"lex": "designs", "type": "nb", "subreddit": "Coronavirus", "vec": [0.5130522847175598, 0.032669272273778915, -0.9251394867897034, -1.1602176427841187, -1.7520135641098022, 0.3188759386539459, -0.5406146049499512, 0.05663304403424263, -0.744211733341217, -1.0573781728744507, 0.43476054072380066, 0.8296738862991333, 2.853811025619507, -0.5145760178565979, -1.407889723777771, -2.2022757530212402, -1.1020644903182983, -0.22681906819343567, 0.49209025502204895, -0.7525967359542847, -1.6543083190917969, -0.7809067368507385, -2.1689870357513428, 0.37176868319511414, 2.81500506401062, 0.5851560235023499, -1.3072311878204346, 0.23833924531936646, 0.041200824081897736, -0.5981003642082214, 0.49647459387779236, 1.9514356851577759, 0.5081503391265869, 0.56309574842453, 0.7368130087852478, -0.06938531994819641, -0.9119196534156799, -0.5432572364807129, -0.615388810634613, 0.8456119894981384, 0.012688404880464077, 0.04315797612071037, 0.7228914499282837, -0.5112990140914917, 0.3212086260318756, 0.014764907769858837, 0.11737268418073654, -0.9086834192276001, -0.6928709745407104, 1.198259711265564, -1.8350498676300049, -0.27790549397468567, -1.1901063919067383, 2.4128761291503906, -1.8135477304458618, 0.43718770146369934, -0.3110085129737854, 0.7463264465332031, 0.17274367809295654, -1.5689220428466797, -0.10198245942592621, -0.3305739164352417, 1.7370190620422363, 1.518546462059021, 0.8249363899230957, -0.20161278545856476, 0.20261582732200623, -2.230329990386963, -1.9490549564361572, -0.8301651477813721, 0.18891358375549316, 0.1353292465209961, -0.28486382961273193, 1.7088004350662231, -0.02937011606991291, -2.367006301879883, -0.41700083017349243, 1.7093549966812134, 0.2868190407752991, 2.0068717002868652, 0.389003187417984, 0.39477628469467163, -1.283323884010315, -0.7075607180595398, -1.5656816959381104, -1.3533402681350708, -0.8537787199020386, 1.2606700658798218, -0.6041529178619385, 1.8083242177963257, -1.1836304664611816, -2.3001785278320312, 0.6525187492370605, 0.21953360736370087, 0.2093176692724228, -2.0327939987182617, -0.16721004247665405, -0.06602233648300171, -0.1076812818646431, -1.084132432937622, -0.4596092998981476, 1.1345336437225342, 0.6414083242416382, -1.5305997133255005, -0.8177953958511353, -0.6060707569122314, 0.1721034198999405, -0.6888663172721863, -0.11252819746732712, -0.3479744791984558, 0.5986495614051819, 0.7682466506958008, 0.8155062198638916, -1.616208791732788, -0.5417396426200867, -1.242114543914795, -0.6112856864929199, 0.6232796907424927, 1.1399732828140259, 0.09641197323799133, 2.6202402114868164, -0.9659251570701599, -1.4723536968231201, -2.132892370223999, -0.6450725793838501, 2.841001272201538, -0.21671804785728455, -1.763944149017334, 2.735980987548828, 0.7896576523780823, 0.4660555124282837, -2.1179587841033936, -1.1129984855651855, -2.0809664726257324, 1.159538745880127, 0.892238438129425, -1.7631289958953857, -0.5655015110969543, 1.9106218814849854, 0.23082198202610016, 2.2400505542755127, -0.900923490524292, 2.2580745220184326, 0.49425676465034485, -0.11354978382587433, 0.5022718906402588, -2.1611390113830566, -1.4637748003005981, 0.27456167340278625, -0.01596483774483204, 0.08586253225803375, -1.207788109779358, 1.946336030960083, 0.3470644950866699, -1.2195079326629639, -0.05321582034230232, 1.0863531827926636, 0.8282997608184814, -0.2928053140640259, 0.07669338583946228, -0.41999706625938416, 0.4759329557418823, 1.0325839519500732, -1.8904356956481934, 1.8957923650741577, 1.618273377418518, -0.7135826945304871, -2.0716910362243652, -0.9105675220489502, 1.037534475326538, 1.3314145803451538, 0.7741562128067017, -0.18376308679580688, 0.36980539560317993, -0.5473322868347168, -0.7423994541168213, -2.0403993129730225, 1.45079505443573, 3.1584701538085938, -0.11966662853956223, -0.5059592723846436, 2.6745922565460205, 0.7625691890716553, -0.20800700783729553, -2.899958372116089, 0.8536426424980164, 0.5874103903770447, -0.5287394523620605, 0.5022848844528198, -2.2898128032684326, 0.9352908730506897, -0.280057430267334, -0.4507579207420349, -0.8370507955551147, -1.0770050287246704, -1.7066861391067505, -1.1421699523925781, 0.1169888824224472, 0.9803758263587952, 0.7370300889015198, -1.21811044216156, -1.147508978843689, 0.2835407555103302, -1.065436601638794, 1.3656126260757446, 0.15276826918125153, -1.6523081064224243, 1.3996654748916626, 0.765929639339447, 2.435053586959839, -0.7718227505683899, 0.24850791692733765, 0.05261415243148804, -0.8847147822380066, 1.7117863893508911, 0.6497103571891785, -1.5494886636734009, 0.23331418633460999, -1.3667036294937134, 0.11890697479248047, -0.9029046893119812, 1.053719401359558, -0.8087272644042969, 0.4444679319858551, -0.6043990850448608, -1.9125468730926514, 0.8788084387779236, 0.9381486773490906, -2.2403206825256348, -2.163325309753418, 0.29609623551368713, -0.8616282939910889, -1.0444917678833008, -0.26943233609199524, 0.3897160291671753, 0.39599111676216125, -0.19210724532604218, 3.026553153991699, -0.7145481109619141, 0.9538141489028931, -1.366883635520935, -0.13144277036190033, -0.9880319833755493, -1.1762615442276, -0.273355633020401, -3.0641329288482666, 1.72431218624115, -0.2104223519563675, 0.25936585664749146, 2.7054245471954346, 1.5124176740646362, -1.1526668071746826, 1.310218334197998, 0.4840715527534485, -1.2141962051391602, 0.2922901213169098, -1.9484851360321045, -1.7908350229263306, 0.09078838676214218, -1.553096890449524, -0.6224895119667053, 1.2737579345703125, -0.9684776067733765, 0.0780201330780983, 0.054651711136102676, -0.9286374449729919, 0.2682158350944519, 0.4819532632827759, -0.1501529961824417, 0.34235888719558716, 0.6476472616195679, -0.010587857104837894, -0.7224072813987732, -0.21933935582637787, 2.7351911067962646, 1.7272403240203857, 0.43908292055130005, 1.1420438289642334, 0.9486563205718994, -1.6390793323516846, -0.1963193565607071, 0.48375698924064636, 0.09530287235975266, 0.6625668406486511, -0.1836143583059311, 0.8412240743637085, -2.2220282554626465, 0.7569908499717712, -0.3252888023853302, -0.7301543951034546, -0.8423237800598145, -0.8501970171928406, 0.8370763659477234, -0.8646007180213928, 0.14642122387886047, -1.2763859033584595, -1.7919467687606812, -0.4598933756351471, -1.443068027496338, 0.8689407110214233], "sim": 0.38796335458755493, "freq": 343.0, "x_tsne": 14.39687442779541, "y_tsne": 21.197154998779297}, {"lex": "supercomputers", "type": "nb", "subreddit": "Coronavirus", "vec": [0.15481367707252502, 0.015649640932679176, 0.2501235902309418, -0.7157526016235352, -0.17782849073410034, 0.2816917300224304, 0.6190696358680725, -0.024292152374982834, -0.0563616007566452, -0.49866706132888794, 0.3310069441795349, 0.2461410015821457, 0.49776309728622437, 0.3529174327850342, 0.19885411858558655, 0.5988106727600098, 0.1701221913099289, 0.28367897868156433, 0.9508777856826782, 0.02187255769968033, -0.313764363527298, 0.32380276918411255, 0.3102019727230072, -0.03473592549562454, 0.7562116384506226, -0.024177903309464455, 0.12392599880695343, -0.37936973571777344, 0.5698226094245911, 0.22683724761009216, 0.5690940022468567, -0.10729259997606277, 0.403022825717926, 0.1840568631887436, -0.06478831171989441, 0.7330136299133301, -0.026376493275165558, 0.3459959328174591, -0.009568333625793457, -0.16380372643470764, -0.1353243589401245, -0.06563223898410797, -0.2317267656326294, -0.11925237625837326, 0.07419567555189133, -0.14357876777648926, 0.33723437786102295, 0.01588381454348564, -0.553382396697998, 0.17489561438560486, -0.0610591284930706, -0.11411484330892563, -0.1099531352519989, 0.3158566951751709, -0.22588810324668884, 0.1923178881406784, -0.48983341455459595, 0.3438981771469116, -0.21337951719760895, -0.30810990929603577, -0.15778127312660217, -0.012488923035562038, 0.25950050354003906, 0.34651243686676025, -0.5207822918891907, 0.1333065629005432, 0.3198964297771454, -0.2449159324169159, 0.24449573457241058, -0.33209705352783203, 0.2788154184818268, 0.47239112854003906, -0.35838431119918823, 0.30095306038856506, 0.1843833476305008, -0.38652104139328003, -0.1503874808549881, 0.5649583339691162, 0.09915787726640701, 0.46888267993927, -0.2113572061061859, -0.23866841197013855, -0.8575541377067566, -0.10470398515462875, -0.034113287925720215, -0.10969839990139008, -0.580676794052124, -0.4893341064453125, 0.8585958480834961, 0.1838335543870926, 0.5884415507316589, -0.052932776510715485, 0.3312738537788391, 0.13630633056163788, -0.5455374121665955, -0.5800716280937195, -0.021262360736727715, 0.38644930720329285, 0.5081605315208435, -0.08287381380796432, -0.4467345178127289, -0.0011942916316911578, 0.11924630403518677, -0.2946230173110962, 0.18345953524112701, -0.23579546809196472, 0.14576220512390137, 0.31154322624206543, -0.12600389122962952, -0.17246751487255096, -0.7885711789131165, -0.31851643323898315, -0.2187601774930954, 0.04345175251364708, -0.44633975625038147, 0.03966907411813736, -0.5546373724937439, -0.6336925029754639, -0.25618216395378113, -0.40627485513687134, -0.06857619434595108, -0.09694348275661469, -0.2400730848312378, 0.32384440302848816, 0.43856218457221985, -0.500557541847229, 0.35661235451698303, -0.3637910485267639, 0.41025346517562866, 0.004921701271086931, -0.36121729016304016, -0.14240868389606476, 0.33259469270706177, -0.6414124369621277, 0.47157955169677734, 0.3174313008785248, -0.560324490070343, -0.07457175105810165, -0.46224066615104675, -0.2321242094039917, 0.15157797932624817, -0.6140205264091492, 0.06861451268196106, -0.364871621131897, 0.264507532119751, 0.08027250319719315, 0.044252485036849976, 0.04335925728082657, 0.5529317259788513, 0.4639061391353607, 0.7897393703460693, -0.5972433686256409, 0.17165987193584442, -0.0670291930437088, 0.12615592777729034, -0.10870932042598724, -0.1766539067029953, 0.1203342080116272, -0.3879711925983429, -0.5037135481834412, 0.3319302797317505, -0.3042597770690918, 0.3749037981033325, -0.3884534537792206, -0.02547842636704445, -0.16889050602912903, 0.43720316886901855, -0.03613458201289177, 0.08260911703109741, 0.3511618971824646, 0.058817457407712936, -0.20594193041324615, 0.13852787017822266, 0.5027469992637634, 0.053487714380025864, -0.9301542043685913, -0.5007572770118713, 0.2169773131608963, 0.14686144888401031, -0.5704323053359985, -0.02354614809155464, 0.6132413148880005, 0.4021017253398895, 0.16103686392307281, -0.29142066836357117, -0.012606152333319187, 0.8102850317955017, 0.19669434428215027, 0.3049198091030121, -0.3024364113807678, -0.4886341989040375, -0.33841440081596375, -0.7736076712608337, -0.050701480358839035, -0.2335861176252365, -0.4193199574947357, -0.2118513137102127, -0.05564839765429497, 0.2713644206523895, -0.19977623224258423, -0.5787354707717896, 0.17525796592235565, 0.4974175691604614, 0.24429620802402496, -0.6022691130638123, 0.27469703555107117, -0.43266430497169495, 0.2618701159954071, 0.4192006587982178, 0.20245368778705597, -0.5050731301307678, 1.0842090845108032, -0.3330078721046448, -0.3192692697048187, 0.6090772747993469, -0.3644986152648926, -0.428528368473053, -0.7370942234992981, -0.6826671957969666, -0.7553017139434814, 0.30187925696372986, 0.4767134189605713, 0.08406071364879608, -0.12519870698451996, 0.11315537244081497, -0.2208176553249359, 0.604708194732666, 0.22514690458774567, -0.14156463742256165, -0.11944688111543655, -0.16535088419914246, -0.017763938754796982, -0.39026519656181335, -0.2385982722043991, -0.18813800811767578, 0.1440826654434204, -0.19813433289527893, 0.8592144846916199, -0.02814468741416931, -0.2332785278558731, 0.5827261209487915, -0.3445618152618408, -0.395279198884964, 0.39122653007507324, -0.16072210669517517, -0.021807551383972168, 0.28474125266075134, 0.4104367196559906, 0.07492479681968689, 0.4522014260292053, 0.10958165675401688, -0.13704431056976318, 0.8819091320037842, -0.06572640687227249, 0.1685187667608261, -0.14227722585201263, 0.03219062089920044, 0.3054284155368805, -0.7060949206352234, -0.45396193861961365, -0.38919734954833984, -0.25880300998687744, -0.284945011138916, 0.16554215550422668, -0.15235109627246857, -0.31354010105133057, -0.1379786878824234, 0.17294859886169434, -0.026228472590446472, -0.47861307859420776, 0.3172672390937805, -0.019544392824172974, -0.14519818127155304, -0.4011712968349457, 0.12474116683006287, 0.6223983764648438, -0.5005183815956116, 0.4416755735874176, -0.262084037065506, -0.14660419523715973, 0.5910207629203796, -0.4501749873161316, -0.6098232865333557, 0.1972823292016983, 0.46058735251426697, -0.3079022467136383, -0.33100759983062744, -0.23291601240634918, -0.7946860790252686, 0.3178618252277374, 0.4583665132522583, 0.2541266083717346, 0.5800029039382935, 0.4533659517765045, 0.6275458931922913, 0.6456266641616821, 0.5475409626960754, -0.46091678738594055, -0.01722198724746704, -0.3395903706550598], "sim": 0.38608303666114807, "freq": 71.0, "x_tsne": 13.01940631866455, "y_tsne": 8.086827278137207}, {"lex": "assays", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.70597243309021, -0.43702611327171326, -0.4632490277290344, -0.12436223030090332, 2.1000521183013916, 0.9462306499481201, 0.0078307269141078, -0.8306746482849121, 0.6090587973594666, -0.9950107932090759, -0.07200562953948975, -1.0126808881759644, 1.9379596710205078, -2.4038949012756348, -0.08434408158063889, -1.009169578552246, -1.0243136882781982, -0.854438304901123, 0.4116130471229553, -0.12678976356983185, 0.4794691503047943, 1.2228636741638184, -1.4221925735473633, 0.14209987223148346, 1.5756165981292725, 0.7279753684997559, -1.8546702861785889, -0.413381963968277, 0.35775208473205566, -1.199559211730957, 2.4377875328063965, 0.10495646297931671, 1.1699919700622559, 0.21504224836826324, 1.4844952821731567, 0.16674554347991943, -0.16936300694942474, -0.3925970196723938, -0.4703913629055023, -0.8189448118209839, -0.7266165018081665, 0.23936507105827332, 0.3935067057609558, -1.0971264839172363, 0.5483601093292236, 0.0031875837594270706, 0.3145237863063812, 0.5028014779090881, -1.4588630199432373, -1.2474342584609985, 1.0045268535614014, -0.26162898540496826, -1.4438914060592651, -1.231466293334961, -0.6561864018440247, 0.08482949435710907, -0.07583712786436081, -1.6249980926513672, 0.04192160442471504, -1.789778470993042, 0.1294017881155014, 2.333306312561035, 0.9971449375152588, 1.487169861793518, -1.7788786888122559, -0.34839990735054016, -1.2774362564086914, -0.9974170923233032, -1.0083695650100708, 0.5506976842880249, 1.6774773597717285, 2.157423496246338, 0.2537637948989868, 0.23092281818389893, 0.6557648777961731, -1.5899288654327393, -1.8347604274749756, -0.48068392276763916, -0.6129119396209717, 0.6371892094612122, 0.8590641617774963, 0.640135645866394, -0.4531590938568115, 0.6810138821601868, -0.7521255016326904, -0.23688873648643494, -0.8082860708236694, -0.480946809053421, 0.9032434821128845, 0.7952997088432312, 0.44479259848594666, -0.9831508994102478, 0.05671099200844765, -0.09068115800619125, -0.9252247214317322, -1.4440977573394775, -1.1126220226287842, 1.2546350955963135, 1.9680947065353394, 0.06406546384096146, 0.33194199204444885, 0.6128869652748108, 0.6606695055961609, -0.4132332503795624, -1.8981831073760986, -0.4724045991897583, 1.0529415607452393, 0.15646572411060333, -0.6766869425773621, -0.9525743722915649, -1.0719928741455078, -0.5703126788139343, -1.6767864227294922, -0.7843962907791138, 0.7051061987876892, 0.24101191759109497, 1.4600112438201904, 0.8093306422233582, 0.1451846808195114, 0.161612868309021, -0.9021205902099609, 1.2209588289260864, -1.1970983743667603, -0.18470343947410583, -0.0741407722234726, 1.530514121055603, 0.026087069883942604, 0.21269938349723816, 1.5345126390457153, 0.6963443756103516, 0.3209053575992584, -0.8740936517715454, -0.02562023513019085, -1.3508580923080444, 1.3210878372192383, 0.2622779309749603, -0.9847416877746582, -0.3643259108066559, -0.6310409903526306, -1.047389268875122, 0.8173293471336365, -1.389449119567871, -0.30890893936157227, -0.9197843074798584, -0.31134024262428284, -1.608454942703247, -0.5697121024131775, 0.8807740211486816, 0.4810252785682678, 0.9711205959320068, -0.584130048751831, -0.6427962779998779, 0.8456021547317505, -0.15782031416893005, -0.9276119470596313, -0.8891435265541077, 0.1524207442998886, -0.8676392436027527, -1.5954710245132446, -0.7371185421943665, -0.5624277591705322, 0.09728488326072693, 0.5124850273132324, 0.17950835824012756, 1.1506085395812988, -0.027596956118941307, 1.1094406843185425, 0.19432781636714935, -0.4391348958015442, -0.019310299307107925, 0.10933877527713776, -1.1854252815246582, -0.0929412692785263, 0.46253907680511475, -0.7866862416267395, -1.8652714490890503, -1.3358111381530762, -0.028604401275515556, 0.1997557431459427, -1.1020152568817139, -1.836235523223877, 1.138377070426941, -0.5545330047607422, 0.8177390694618225, -1.5286755561828613, 1.8509423732757568, 0.7453354597091675, 0.7474004030227661, -1.1025445461273193, -0.021239005029201508, -0.2729564905166626, -0.1709984540939331, -1.5691064596176147, 0.29880401492118835, 0.25818932056427, -0.12562713027000427, 0.3704770803451538, 0.1842612624168396, 2.8100550174713135, -1.3987592458724976, -2.400695323944092, 0.6046263575553894, 0.27337196469306946, 0.4548216164112091, -2.0175678730010986, 1.1773972511291504, -1.5103639364242554, 0.12870728969573975, 1.5836613178253174, 0.9114214777946472, -0.9873531460762024, -0.8619554042816162, -0.7485246658325195, -0.2867225706577301, 0.8942688703536987, -0.5918383598327637, -0.48006588220596313, -0.15527212619781494, -2.0574333667755127, -0.6679610013961792, 1.42082941532135, 1.7876149415969849, 1.4486887454986572, -0.06179779767990112, -0.4514378011226654, -0.5912144780158997, 0.6967100501060486, 1.0495706796646118, -0.19487708806991577, -1.1659228801727295, -0.727293074131012, -1.207024335861206, -0.9780080914497375, -0.793590247631073, 0.2587295472621918, -0.8220189213752747, 0.6106794476509094, 2.581092357635498, -1.2535426616668701, -0.05956313759088516, 0.3789916932582855, -1.1941463947296143, -0.3099534511566162, -1.1607997417449951, -1.6606086492538452, -2.008063793182373, 1.0531085729599, 1.9795300960540771, 0.3245331943035126, 0.24595050513744354, 1.164528250694275, -0.37717533111572266, 1.0273436307907104, -0.2662459909915924, -0.8287786841392517, -0.43054884672164917, -0.2593173682689667, 0.3686825633049011, -0.21365834772586823, -1.2691370248794556, -0.2457146793603897, 0.28040140867233276, -0.8527622222900391, 0.6514440774917603, 1.2780171632766724, -1.010926604270935, -0.11771329492330551, -2.3628413677215576, -0.009092044085264206, 0.4673102796077728, -0.3297593593597412, -0.42962750792503357, -1.530472993850708, 0.20199556648731232, 0.1293330043554306, 2.041064739227295, -1.267506718635559, 0.7233362197875977, 0.5214462876319885, -0.6951600313186646, 0.8927726745605469, -0.51677006483078, -2.4391770362854004, -0.44069138169288635, 0.694691002368927, -0.38518062233924866, 1.3851206302642822, -1.7787573337554932, -1.4236326217651367, 0.19196613132953644, 0.4087221026420593, -0.30545827746391296, 0.5411453247070312, -0.13196423649787903, -0.0704488530755043, -0.2996358275413513, -0.4566776156425476, -0.9568095207214355, 1.1069841384887695, -0.38824158906936646], "sim": 0.3845449984073639, "freq": 115.0, "x_tsne": 11.342057228088379, "y_tsne": 24.557695388793945}, {"lex": "monoclonal", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.7199099659919739, -1.683725118637085, 0.8994544148445129, -2.0611631870269775, 0.47950705885887146, -0.23853254318237305, 0.692051887512207, 0.8786845207214355, 0.19863778352737427, -0.0506729781627655, 1.5133758783340454, 1.1324996948242188, 2.4482719898223877, -1.7572524547576904, 2.3806278705596924, 0.12049918621778488, 0.009216305799782276, -0.509320080280304, 2.760281562805176, -1.3183715343475342, -0.885894238948822, 0.46360525488853455, -0.43878039717674255, 1.2006328105926514, 1.1739165782928467, 0.014592236839234829, -1.0204265117645264, 0.7338782548904419, 0.3919880986213684, -1.035770297050476, 3.075061321258545, -0.03401434049010277, 0.8427152037620544, -0.8709263205528259, 0.6875594854354858, 1.3968316316604614, -2.8614354133605957, -0.3780613839626312, -0.029768995940685272, -1.5854637622833252, -0.5881627202033997, 0.7844311594963074, -0.015212462283670902, 1.0535715818405151, 1.5417588949203491, -1.4161794185638428, 0.709397554397583, 2.3356850147247314, -0.4531645178794861, -1.8967770338058472, 1.3979939222335815, 0.015002337284386158, -1.7092317342758179, -0.9199792742729187, -1.1682653427124023, 1.1078181266784668, -0.6731175184249878, 0.8899498581886292, 0.8517106771469116, -3.8835911750793457, -0.40086308121681213, 4.110195159912109, -1.3809630870819092, 1.6239013671875, -1.3175345659255981, 0.36806628108024597, -0.6609029173851013, 1.5616925954818726, -0.3861088752746582, 0.44620877504348755, -1.6243135929107666, -1.464648962020874, -0.8159433007240295, 0.9024497866630554, 0.09142112731933594, -0.47407564520835876, -0.19300079345703125, 0.07512356340885162, 1.6591864824295044, -0.5231069326400757, -1.4582017660140991, -1.1316832304000854, 0.2854606807231903, -0.5258802175521851, 0.5512205958366394, -0.19403287768363953, -1.2417889833450317, -0.16116437315940857, 0.8891732096672058, 2.071507453918457, -1.382079839706421, -1.694054365158081, 0.5969396829605103, -1.6289902925491333, -0.5785132646560669, -1.606357455253601, 1.686809778213501, 0.5114188194274902, 0.22009365260601044, -0.35687437653541565, 0.8303964734077454, 0.6837736964225769, -0.30908921360969543, -0.350974977016449, -1.7987589836120605, -1.74574613571167, 0.6925868391990662, -1.4876394271850586, 1.7082048654556274, -0.5453936457633972, 0.11721184104681015, 1.3545644283294678, -0.09288054704666138, -1.3403154611587524, -0.43381860852241516, -0.8611806631088257, 1.01107656955719, 1.184949517250061, 1.9473153352737427, -0.7801773548126221, -1.534460425376892, -0.207340270280838, -0.8897272348403931, 1.1131596565246582, -1.0560799837112427, 0.1814972460269928, -1.1169261932373047, -0.7332144379615784, -1.2826272249221802, -0.7027546167373657, 0.5000705718994141, -1.1618980169296265, -1.0414488315582275, 0.6806408166885376, 1.5538119077682495, -2.072004556655884, -0.05920032039284706, -0.6409017443656921, -1.0060219764709473, -0.09540347754955292, -0.1261964589357376, -3.0998964309692383, -0.7442458271980286, 0.011512272991240025, 0.46900638937950134, -1.3324103355407715, -2.1176717281341553, -2.375718832015991, 0.260341078042984, -0.05090494453907013, 1.5355654954910278, -3.5066781044006348, 0.32365700602531433, 2.4906718730926514, -0.004304671194404364, -0.8388893008232117, 1.382861852645874, 0.035441864281892776, -0.80645751953125, 1.2835109233856201, -0.5834454298019409, 0.5583901405334473, -0.7186721563339233, -0.2214023470878601, -1.7736228704452515, -0.2163352072238922, 0.48985135555267334, 1.2026960849761963, -1.4694247245788574, 1.5939292907714844, -0.2226880043745041, -2.0647716522216797, 0.1285707950592041, 0.8371851444244385, -0.7669695615768433, -0.8730530738830566, 0.6640598773956299, 0.15153023600578308, -0.8237828612327576, -1.6839842796325684, -1.289963722229004, 1.4653466939926147, 0.17138248682022095, -0.17813852429389954, -0.20017705857753754, 1.0915758609771729, 1.1371536254882812, 1.1786340475082397, -0.9107692241668701, -1.1573982238769531, -2.1226706504821777, -1.9316173791885376, -1.6191434860229492, -0.07525806874036789, -0.32869789004325867, -0.5356035232543945, 0.5347210764884949, 1.408922553062439, 0.9355062246322632, -0.8036168813705444, -4.757900238037109, -0.6844845414161682, -1.2709276676177979, 2.43235445022583, -1.331784963607788, 0.29866424202919006, -0.6187394857406616, 1.791460394859314, 1.0262258052825928, 0.6322392225265503, -1.37763512134552, -1.3079029321670532, -1.1428799629211426, -0.8403247594833374, 2.0942564010620117, -0.5978128910064697, 0.2833690345287323, -0.006087237969040871, -1.9364385604858398, -0.42306309938430786, 0.6354050040245056, 1.130556344985962, 0.043473802506923676, -0.8308435082435608, 1.0975536108016968, -1.545135259628296, -0.6966480016708374, 1.500766634941101, -0.5167360305786133, 0.7425203323364258, -0.0771850198507309, -0.5033960342407227, 0.9946423172950745, 1.2879823446273804, 1.0511794090270996, 1.154120683670044, -0.297055184841156, 1.2245287895202637, 0.11839864403009415, -0.92011559009552, 0.47093626856803894, -1.5046072006225586, -0.6177732348442078, -0.27844950556755066, -1.1874288320541382, -0.7544079422950745, 0.7457565665245056, 3.1476480960845947, -0.5507712364196777, 0.4476192593574524, 1.4124534130096436, -0.17535607516765594, 0.8956105709075928, -0.4496585726737976, -2.3820717334747314, 0.13501077890396118, -2.5075957775115967, 0.126422718167305, 0.9889113306999207, -1.264101266860962, 0.3958660066127777, -1.3952418565750122, -0.6058483719825745, 0.40352559089660645, -0.8131616115570068, -1.7833093404769897, -0.9480616450309753, -0.31858402490615845, 1.2584854364395142, -0.19141939282417297, 1.0067676305770874, 0.3921518623828888, -0.05040353909134865, -0.30988770723342896, -0.6795743703842163, 0.7444984912872314, -2.4252219200134277, 2.1312549114227295, 0.13074122369289398, -1.8693102598190308, 0.7203821539878845, -0.3033199608325958, -1.1508666276931763, -2.0510592460632324, 0.8884748816490173, 1.9017531871795654, 1.5457000732421875, -1.231202244758606, -0.8692334294319153, -0.1898454874753952, 2.6197216510772705, -2.3853580951690674, 0.7805249094963074, 1.3677386045455933, 0.21964102983474731, -0.5151920318603516, -0.012229816056787968, 0.6927227973937988, 1.127987027168274, 0.9768581986427307], "sim": 0.3819161355495453, "freq": 214.0, "x_tsne": -5.280434608459473, "y_tsne": 16.96754264831543}, {"lex": "strategies", "type": "nb", "subreddit": "Coronavirus", "vec": [-0.2184787392616272, -1.6912331581115723, 2.2269062995910645, -0.330337256193161, 0.0007594064227305353, -0.0654527097940445, 0.1960332840681076, -1.3772870302200317, -1.4073923826217651, -1.3509788513183594, 1.0635353326797485, -0.1886880248785019, 2.49159836769104, -0.958528459072113, 0.8409404754638672, 0.07083184272050858, -1.0834920406341553, -1.5434443950653076, -0.47315776348114014, -0.1028745099902153, 0.8384193778038025, 0.7031963467597961, -1.679356336593628, -0.6779654622077942, 2.3339762687683105, 1.9057713747024536, 0.8951786756515503, -0.9000349044799805, 1.0111404657363892, 1.8777165412902832, 0.3418741524219513, 0.24359583854675293, -1.8531757593154907, -0.015160856768488884, -0.6685584783554077, -0.27579188346862793, -0.2687320113182068, 0.15017403662204742, -2.1614019870758057, 1.280672550201416, -3.3591861724853516, 0.5857807993888855, -1.6813430786132812, -1.1470928192138672, 2.617544412612915, 0.8152111172676086, 2.5429282188415527, -1.0116791725158691, -3.5542006492614746, -1.672099232673645, -0.7161084413528442, -1.8135464191436768, -1.8798600435256958, -0.37890809774398804, -1.85051691532135, 1.2403264045715332, 1.3839892148971558, -1.5653883218765259, 0.6455932855606079, 0.6733781695365906, 1.8198896646499634, 1.4900710582733154, 0.5458482503890991, 2.1277248859405518, -2.1135988235473633, -1.8037482500076294, -0.0031521711498498917, -0.2641901969909668, -1.3421194553375244, -1.8432559967041016, -2.068821430206299, 0.7685313820838928, 1.6808197498321533, 1.3641642332077026, -0.41979384422302246, -1.8714656829833984, -1.7132350206375122, 2.2240419387817383, -2.829287528991699, 1.5268093347549438, 3.4659793376922607, 1.480677843093872, -1.8636457920074463, -1.3555569648742676, -1.118868112564087, 1.0919857025146484, -0.22834643721580505, 2.3900463581085205, -1.9867219924926758, 0.2511550784111023, -1.3241690397262573, -2.6888906955718994, 2.039367437362671, 2.1691501140594482, -2.7192931175231934, -2.1693942546844482, -1.7782671451568604, -0.8341039419174194, -0.492876797914505, -3.8524975776672363, -0.9589847922325134, 0.2023424357175827, -0.11614005267620087, 1.9686695337295532, 1.008676290512085, -1.5006935596466064, 2.3637194633483887, -1.755275011062622, -0.29079338908195496, -1.0718694925308228, 2.8437180519104004, 1.5157421827316284, -1.7633365392684937, -0.0825214758515358, 0.4258700907230377, -1.1308503150939941, 1.2488062381744385, 0.6864528656005859, 0.28272780776023865, 1.9423919916152954, -0.2094181776046753, -1.330739140510559, -2.3304996490478516, -1.952973484992981, 0.6510646939277649, 2.5777571201324463, -0.12105339765548706, -0.8873232007026672, 2.6165406703948975, -1.0565482378005981, -1.5076791048049927, 0.4113274812698364, -0.6721349954605103, -1.7431178092956543, 0.8455603122711182, 1.162729263305664, -1.3853349685668945, 0.5053077936172485, 0.9734238982200623, -1.3314992189407349, 0.6681867241859436, 0.4510132074356079, -1.9994980096817017, -1.541728138923645, 0.951110303401947, -0.9242188334465027, 4.035707950592041, 1.5021792650222778, -0.6441284418106079, -3.1588668823242188, 0.5493952631950378, 0.11999158561229706, 1.8952076435089111, -0.7896960377693176, 0.3341423571109772, 0.38688650727272034, -0.5976327657699585, -1.820781946182251, -0.3003050982952118, -2.4389352798461914, 1.8440685272216797, -0.06823812425136566, 1.3307677507400513, 0.8543434143066406, 4.124329090118408, 0.9902161359786987, 0.7313640713691711, 1.0463299751281738, -0.8286991715431213, -0.5592174530029297, -1.5170525312423706, 0.6808128952980042, 0.46614494919776917, -1.7996108531951904, 0.6914569139480591, -3.341359853744507, -2.2085561752319336, 2.1478326320648193, 1.4118785858154297, -0.6674160957336426, 0.6798276305198669, 1.6171352863311768, -1.5319608449935913, 0.4634377062320709, -2.0491669178009033, 0.632311224937439, 1.6313709020614624, -1.651405930519104, -2.7312722206115723, 0.06121385470032692, 0.0331902839243412, -0.6041814088821411, -1.637198567390442, -1.9507476091384888, -0.14496254920959473, -2.4701125621795654, -0.041156526654958725, 2.0530803203582764, -0.9101154208183289, -0.131450816988945, -0.7325167059898376, -0.611396074295044, 0.414066344499588, 0.03512006253004074, -3.400355100631714, 1.7427743673324585, -0.2726525068283081, 0.5475472807884216, -1.1972821950912476, 2.252641201019287, -1.9449114799499512, -0.5095543265342712, -1.581828236579895, 1.0089972019195557, -0.5983319878578186, 0.16947557032108307, -0.7407926917076111, 1.2815015316009521, 0.20900650322437286, -2.6066629886627197, -0.44091424345970154, 0.6351731419563293, 0.9144030213356018, 0.35826361179351807, 1.7518452405929565, -1.5330908298492432, -0.5133695602416992, 0.004712732043117285, 0.8014655113220215, -1.6717106103897095, 1.1652015447616577, 0.020813222974538803, 0.9862097501754761, -2.552875518798828, 0.36825454235076904, -0.5112794041633606, 0.045309267938137054, 4.339357852935791, -1.091590166091919, 0.8996882438659668, -2.745911121368408, -2.0987935066223145, -1.0975687503814697, -2.1199491024017334, -1.0301388502120972, -0.9438347816467285, 2.8057119846343994, -0.7121062278747559, 2.542958974838257, 1.2750036716461182, 1.9522360563278198, -2.7430732250213623, 0.9104448556900024, 1.412315845489502, -1.4708449840545654, 1.5938560962677002, 0.2091371864080429, -0.2465236335992813, 0.8169840574264526, -1.4135342836380005, -2.3516435623168945, -0.07647497951984406, 0.6945200562477112, 0.29134029150009155, 0.03608857840299606, 0.33831456303596497, 1.1030526161193848, 0.3731769919395447, -0.5859097838401794, 0.46886587142944336, 0.04204946383833885, -0.16009779274463654, -1.8252465724945068, 0.40688204765319824, 0.9459934234619141, 1.1189563274383545, -0.4662575423717499, 2.841745138168335, -0.6968936920166016, -1.236376166343689, -1.3262076377868652, 1.913158893585205, -2.729477882385254, 0.5835627317428589, 1.1056808233261108, -1.351602554321289, -0.7082275152206421, 0.8253505825996399, -1.551126480102539, 2.361844062805176, -1.2165457010269165, -0.20964916050434113, 2.3002490997314453, 1.8683760166168213, 1.0160019397735596, -1.8041712045669556, -0.8743468523025513, -1.502475380897522, -0.1582438200712204, -3.3958239555358887], "sim": 0.3794310688972473, "freq": 1362.0, "x_tsne": 11.2003812789917, "y_tsne": 33.16389465332031}, {"lex": "subtypes", "type": "nb", "subreddit": "Coronavirus", "vec": [0.8826078772544861, -0.13443729281425476, 0.3790383040904999, -0.4677797853946686, 0.43396154046058655, -0.7323474287986755, 0.5702304840087891, -0.5604313611984253, -0.8326252698898315, -0.6599236726760864, 0.25412383675575256, -0.4663560092449188, 0.817499577999115, 0.29244929552078247, -0.06269615143537521, 0.12270372360944748, 0.13077868521213531, -0.6661832332611084, -0.12139362841844559, 0.5166596174240112, -0.6279053092002869, 0.15710552036762238, 0.2049357295036316, -0.41470205783843994, 0.4857900142669678, 0.17037588357925415, -0.1714334338903427, -0.19553600251674652, -0.5270277857780457, -1.006197452545166, -0.5640901923179626, -0.9906725883483887, -0.9842427968978882, -0.7945179343223572, 0.02327430620789528, 0.2232329547405243, 0.0014710698742419481, -0.41099321842193604, -1.1563122272491455, -0.25289589166641235, 0.1352812945842743, 0.08610878139734268, 0.1259336918592453, -0.04752534255385399, -0.058675624430179596, -0.24006441235542297, 0.3849794566631317, 0.04736728221178055, -0.5704361200332642, -0.4911893308162689, 0.07924479991197586, -0.7953779101371765, -0.7239206433296204, -0.20699356496334076, -0.36006805300712585, 0.359781414270401, -0.20735731720924377, -0.41602426767349243, 0.7027100324630737, -0.8393997550010681, -0.13294468820095062, 0.35593459010124207, -0.010400569997727871, 0.05281570926308632, -0.2990286946296692, 0.23155477643013, 0.21902897953987122, 0.15735778212547302, -0.8373186588287354, 0.16844144463539124, 0.9112319946289062, -0.9464166760444641, 0.2678835391998291, 0.09351280331611633, 0.2910926938056946, -0.27858686447143555, -0.19431011378765106, 0.3999249339103699, -0.31913942098617554, 0.9983758926391602, 0.2294427454471588, -0.7568912506103516, -0.23210009932518005, 0.33200281858444214, -0.13077974319458008, 0.0048997895792126656, -0.7136667966842651, -0.31392866373062134, 0.014197670854628086, -0.392042338848114, -0.523673951625824, -0.04986782744526863, -0.06938238441944122, 0.0735548585653305, -0.5195195078849792, -0.17876261472702026, -0.1419779658317566, 0.08997452259063721, 0.8825652003288269, -0.4058915972709656, -0.1995178610086441, 0.17087647318840027, -0.21214427053928375, -0.10351860523223877, -0.4654523432254791, 0.34122154116630554, 0.3880816102027893, 0.2779988944530487, 1.1631687879562378, -0.4004805386066437, 0.4980580806732178, -0.15142282843589783, -0.4728658199310303, 0.03127064183354378, -0.478630006313324, 0.2160099297761917, -0.35860931873321533, -0.04991530999541283, 0.9392675161361694, 0.048683565109968185, -0.26531627774238586, -0.2575003504753113, -0.123052679002285, -0.18603269755840302, 0.5529110431671143, 1.4143637418746948, -0.014596671797335148, -0.13214869797229767, 0.012585869990289211, 0.5856737494468689, -0.27573564648628235, -0.9364036321640015, -0.548564076423645, -1.120435118675232, 0.2928829789161682, 0.4782448410987854, -0.20148047804832458, -0.08462712168693542, -0.16709177196025848, -0.5386524796485901, -0.7677123546600342, -0.4606907069683075, -0.4298132359981537, -0.649071455001831, 0.013373425230383873, 0.161391019821167, -0.2190643697977066, -0.616233766078949, 0.5027148127555847, 0.05058334395289421, -0.25057098269462585, -0.3894461691379547, -0.01783078722655773, -0.3510266840457916, 0.03947467729449272, -0.26189228892326355, 1.0919251441955566, -0.058351267129182816, -0.5794917941093445, -0.0366886742413044, -0.6089900732040405, 0.4999743700027466, 0.2837259769439697, 0.593217134475708, 0.3252948522567749, 0.4154563546180725, -0.4574216902256012, 1.007738709449768, 0.11221199482679367, 0.2560821771621704, 0.03828473016619682, 0.1900465041399002, -0.6350388526916504, -0.19517497718334198, -0.3775862455368042, -0.025076914578676224, 0.00940486416220665, -0.4023742377758026, 0.1477746218442917, -0.1161579042673111, -1.3905714750289917, 0.45061832666397095, 0.8163617253303528, 0.6744476556777954, -0.06479185074567795, 0.47394484281539917, 0.4202195703983307, -0.38978224992752075, -0.04829578474164009, -0.4176924526691437, 0.1842351257801056, -0.4206772744655609, 0.07867494225502014, 0.4266347289085388, -0.11072388291358948, -0.5128249526023865, 0.4649472236633301, 0.45729556679725647, 0.5914823412895203, -0.21139700710773468, -0.2988607883453369, 0.8803240060806274, -0.41429516673088074, 0.4151724874973297, -0.2285841554403305, -0.11133035272359848, -0.5024067163467407, 0.391467809677124, 0.33697494864463806, -0.16042958199977875, -0.6731600165367126, -0.7199153304100037, 0.5450665354728699, 0.5683742165565491, -0.08804823458194733, -0.30409982800483704, -0.9150763154029846, -0.8422212600708008, 0.03262840211391449, 0.18614749610424042, 0.592240035533905, 0.4098532795906067, 0.03098520077764988, 0.10195360332727432, 0.09543389827013016, -0.12011782079935074, 0.29692307114601135, 0.8683483004570007, -0.21589834988117218, -0.15075792372226715, 0.08189592510461807, 0.09824707359075546, -0.04529745131731033, -0.021069534122943878, 0.6381623148918152, -0.30359402298927307, -0.04841391742229462, -0.10503915697336197, -0.5141677856445312, 0.09779003262519836, -0.0696527361869812, -0.5770936012268066, -0.0425092987716198, -0.319546639919281, 0.04331905022263527, -0.24531050026416779, 0.04467363283038139, 0.7487775087356567, -0.15890777111053467, -0.24784377217292786, -0.29326725006103516, -0.367256760597229, 0.2151876986026764, 0.1981603354215622, -0.06575094163417816, 0.20506273210048676, -0.07949677854776382, -0.31594568490982056, 0.3464363217353821, 0.14288705587387085, 0.43570515513420105, 0.16126398742198944, -0.5680228471755981, 0.6574317216873169, 0.9277251362800598, 0.1323672980070114, -0.017308227717876434, 0.664832353591919, 0.033613260835409164, 0.47548720240592957, 0.35045474767684937, -0.5839614272117615, -0.8182623982429504, 0.3237314522266388, 0.05250411108136177, 0.5414356589317322, -0.4915047883987427, 1.0749443769454956, -0.4645707309246063, 0.2353770136833191, -0.07271180301904678, 0.379139244556427, -0.3877229690551758, -0.05101195350289345, -0.016286509111523628, -0.391985148191452, 0.03527370095252991, -0.4603656828403473, -0.47490590810775757, -0.42992648482322693, 0.1575319617986679, 0.16289305686950684, 0.45964112877845764, -0.3908429443836212, 0.33387649059295654, 0.015558055602014065, -0.5062382817268372, -0.07759368419647217, -0.6498263478279114, -0.37079182267189026], "sim": 0.37866920232772827, "freq": 27.0, "x_tsne": 6.697515487670898, "y_tsne": 7.959527969360352}, {"lex": "frontrunners", "type": "nb", "subreddit": "Coronavirus", "vec": [0.381797730922699, 0.5145953297615051, 0.9135392308235168, 0.13755296170711517, 0.1301228106021881, 0.862321674823761, 0.5808642506599426, -0.18554332852363586, -0.04419240355491638, -0.9184570908546448, 0.4522930383682251, 0.8060436248779297, -0.24237029254436493, -0.4248887896537781, 0.4957540035247803, -0.7419684529304504, -0.9762541055679321, -0.7303763628005981, 0.6156494617462158, 0.5414934754371643, 0.28025656938552856, -0.26140278577804565, 0.19276753067970276, 0.11442257463932037, -0.36689120531082153, -0.5348061919212341, -0.49969562888145447, 0.15444520115852356, 0.6786710619926453, -0.19073864817619324, -0.5262147784233093, -0.5914037227630615, 0.32748109102249146, -0.5545462369918823, 0.04167468100786209, 0.154504656791687, -0.5803331136703491, 0.6429370641708374, -1.8069769144058228, -0.042824871838092804, -0.4422225058078766, 0.21414749324321747, 0.4826056659221649, -1.4327534437179565, 0.3377300202846527, -0.4530644118785858, -0.028087617829442024, 0.3269437253475189, -0.7926353812217712, 0.9984433054924011, -0.23058439791202545, 0.9997495412826538, 0.08792334794998169, 0.41411736607551575, -0.43702128529548645, 0.008980720303952694, 0.15512390434741974, -0.23778857290744781, 0.29646530747413635, -0.46889421343803406, 0.10286659002304077, -0.5673643350601196, -0.008445424027740955, 0.7118902802467346, -0.18942077457904816, 1.2951140403747559, -0.17728197574615479, -0.20484527945518494, -0.8526060581207275, -0.08447103202342987, -0.5364089012145996, 0.18710912764072418, -0.1301807463169098, -0.40603312849998474, 0.10416047275066376, -0.15801410377025604, 0.47304221987724304, 0.410785436630249, 0.5876866579055786, -0.25705572962760925, 0.04807727411389351, 0.4447662830352783, -1.5185362100601196, 0.3120693564414978, -0.19965405762195587, -0.07926135510206223, -0.04832833632826805, -0.5633585453033447, 0.3283725082874298, 0.17265862226486206, 0.06895732134580612, -0.6891461610794067, 1.1518776416778564, 0.42031195759773254, -0.6578097939491272, -0.6600731611251831, -0.3981803059577942, -0.8408287763595581, 0.0989467203617096, 0.3972965180873871, 0.0917682871222496, 0.7093765139579773, 1.159714698791504, -0.182529479265213, 0.22935263812541962, -0.07751097530126572, 0.6862499117851257, -0.0465017631649971, 0.5647464394569397, -0.3143088221549988, 0.3696223497390747, -0.3905312120914459, -0.7348396182060242, -0.4554751217365265, 0.24546033143997192, -0.5117142796516418, 0.402521014213562, -0.03434617817401886, -0.833320140838623, -0.10099902749061584, -0.8826034665107727, 0.8871661424636841, -0.42165258526802063, 0.3008946478366852, 0.3215343654155731, 1.734513759613037, -0.5734590291976929, -0.43505531549453735, 0.8224162459373474, -0.5086786150932312, 0.1564694344997406, -0.3619861602783203, -1.1297158002853394, -0.26119187474250793, -0.24090167880058289, -0.07411652058362961, -0.31556108593940735, -0.4311182498931885, 0.23289667069911957, 0.41008618474006653, 1.2289093732833862, -0.5498723983764648, 0.5778646469116211, -0.4564589262008667, 1.1486066579818726, -0.38124382495880127, -1.0336041450500488, -0.5037540793418884, 0.3282950222492218, 0.29041001200675964, 0.6751765608787537, 1.2408766746520996, 0.5031152367591858, -0.42584550380706787, 0.8784404397010803, 0.13294468820095062, -0.0027327092830091715, 0.07744454592466354, -1.3675214052200317, 0.4450273811817169, 0.8240241408348083, 0.612765908241272, -0.8537174463272095, 1.0418529510498047, -0.11728185415267944, 0.6743302941322327, -1.0597717761993408, 0.25152677297592163, -0.3415377736091614, 0.03671623021364212, -1.0356303453445435, -0.8439088463783264, -0.5413937568664551, -0.10429461300373077, 0.4458346664905548, -0.40907880663871765, -0.3750890791416168, 0.2802906930446625, -0.2750287652015686, -0.14020821452140808, -0.7269145250320435, 0.31336164474487305, -0.8494876623153687, -0.7290007472038269, -0.9667057394981384, -0.2695995271205902, 0.05917065218091011, -0.45416098833084106, 1.5497629642486572, -0.7397051453590393, -1.3931641578674316, 0.2537098526954651, -0.680585503578186, -0.17789262533187866, 0.050166111439466476, -0.6716306209564209, -0.07340353727340698, 0.07149369269609451, 0.5756470561027527, 0.13906730711460114, -0.7174423336982727, 0.785977303981781, -0.18467983603477478, -0.15359501540660858, 0.07030151784420013, 0.08375216275453568, -0.788023829460144, 0.4447612464427948, -0.26803794503211975, 0.5509264469146729, -0.1846114844083786, 0.4126620292663574, -0.31425192952156067, -0.26425448060035706, 0.23953232169151306, -0.9388689398765564, -0.8576509952545166, -0.1825537085533142, -0.3821694254875183, 0.07381023466587067, 0.9920795559883118, 0.7165798544883728, -0.37769877910614014, -0.5964934825897217, 0.40423211455345154, 0.8474376797676086, 0.42736268043518066, 0.6682709455490112, -0.2132800817489624, -0.9931445121765137, -0.012624725699424744, -0.5805726647377014, 0.7069695591926575, 0.49377763271331787, 0.7524737119674683, -0.33819106221199036, -0.47898897528648376, 0.7758467793464661, -0.2032608538866043, -0.2409234493970871, -0.5709808468818665, -1.0805659294128418, 0.1945384442806244, -0.31796976923942566, -0.4051532447338104, 0.3507760763168335, -0.25549986958503723, -0.06202676147222519, -0.07980871945619583, 0.5125609636306763, 0.5637068152427673, 0.051502425223588943, 0.3071548342704773, 0.6292528510093689, -1.0264655351638794, 0.5386592745780945, -0.27989399433135986, -0.1665673702955246, 0.3713320791721344, 0.18794673681259155, 0.27129441499710083, -0.9295884370803833, -0.7522781491279602, -0.47839802503585815, 0.7738024592399597, 0.2801305949687958, 0.7561313509941101, 1.21658456325531, -0.3695627748966217, 0.3051189184188843, -0.24345669150352478, -0.18910688161849976, 0.002682336838915944, -0.47192323207855225, 0.3050251305103302, 0.9337918162345886, -0.6887089610099792, 0.5488864779472351, 0.20009799301624298, -0.8464773893356323, 1.0168274641036987, 0.3276773691177368, -1.0019301176071167, 0.05114229768514633, -0.501172661781311, -0.1621401011943817, -0.6316940188407898, -0.2286119908094406, 0.047778163105249405, 0.10530579835176468, -0.14457683265209198, 0.3815510869026184, 0.7133527994155884, -0.7434104681015015, 0.2558390200138092, -0.23407402634620667, -0.04144652187824249, -0.30478614568710327, 0.13836660981178284, 0.6023386120796204], "sim": 0.3786624073982239, "freq": 29.0, "x_tsne": 15.60462474822998, "y_tsne": -1.802881121635437}, {"lex": "techniques", "type": "nb", "subreddit": "Coronavirus", "vec": [0.16608157753944397, -0.2138291895389557, 0.2709611654281616, -0.8283525109291077, -1.4479162693023682, 2.5172646045684814, 1.9845385551452637, 0.9426597356796265, -1.089963674545288, -0.7829778790473938, -0.22994786500930786, -1.5402500629425049, 2.8989152908325195, -0.0006055222474969923, -0.9126726984977722, -1.6233735084533691, -0.7283758521080017, -0.04833324998617172, 0.684745192527771, 0.08997683972120285, 1.7167750597000122, 2.5221614837646484, -0.7769469618797302, 0.9347920417785645, 2.932966709136963, 0.4198627769947052, 1.100122332572937, -0.5248357653617859, -0.7005532383918762, -0.9568324685096741, 0.007763143163174391, 0.7352659106254578, -0.10943086445331573, -0.9864819049835205, -1.8216580152511597, -0.14347289502620697, -0.23255348205566406, -0.8123486042022705, -2.193040370941162, -0.033030908554792404, -2.4547958374023438, 1.2491482496261597, -1.3832942247390747, -0.3067590892314911, 1.5413188934326172, 2.5083658695220947, 2.50417160987854, 1.1682673692703247, -2.172045946121216, 0.5564655661582947, -0.5431101322174072, -0.5496323108673096, -1.6145362854003906, 0.8072202801704407, -2.8781213760375977, -0.7294355034828186, 0.7066633105278015, -0.9492413997650146, 0.547558605670929, -0.6079798340797424, 3.8682897090911865, 0.36908939480781555, 1.6850296258926392, 2.0163466930389404, -0.6033890843391418, -0.35616424679756165, 0.9631257057189941, -1.3479677438735962, -2.916735887527466, -0.8143719434738159, -0.5297381281852722, 0.7872965931892395, 3.0413966178894043, 2.0015318393707275, -0.31058579683303833, -2.0505459308624268, -0.13230039179325104, 1.938881516456604, -1.2129812240600586, 2.913661241531372, 2.549701690673828, 0.7434162497520447, 0.09057474136352539, -0.3086622953414917, -2.744704484939575, -0.25791361927986145, 0.027873266488313675, 0.8243990540504456, -1.4816218614578247, 1.2278103828430176, -1.0817134380340576, -2.549833059310913, 1.8000719547271729, 2.9253177642822266, -1.4890437126159668, -0.662990152835846, -0.8721565008163452, 0.11765599250793457, 0.7303353548049927, -2.5383872985839844, -0.2791688144207001, 1.5041694641113281, -0.3039538860321045, 0.8675851821899414, -1.9964144229888916, -1.4679999351501465, 1.4164741039276123, 1.2139500379562378, -0.3043380081653595, -0.6136362552642822, -0.49139276146888733, 1.272475242614746, -2.057429790496826, -1.257766604423523, -0.747661292552948, 0.4913318455219269, 1.5145586729049683, -1.2047494649887085, -0.37004727125167847, 0.19384241104125977, 0.7303060293197632, 0.07766040414571762, -3.0537242889404297, -1.4540969133377075, 0.07584250718355179, 0.8290223479270935, 0.49066418409347534, 0.3158624768257141, 3.2107157707214355, -0.07398910820484161, 0.2736525237560272, -1.8797866106033325, -0.14191946387290955, -0.056088950484991074, 1.4156224727630615, 1.0379661321640015, -0.9136776328086853, -0.9126821756362915, -0.22480225563049316, -0.6962795853614807, 0.7910240292549133, -1.3390973806381226, -0.7138158679008484, -0.7441936135292053, -0.6984635591506958, -1.9591985940933228, 0.872633695602417, 0.7793265581130981, -0.25640833377838135, 0.29229438304901123, 0.5736133456230164, -0.15920299291610718, 3.378995656967163, -0.5762260556221008, 0.49037888646125793, 0.6550662517547607, 1.0986416339874268, -0.4024779498577118, -0.2683931291103363, -2.5408976078033447, 0.2239074856042862, -0.41758841276168823, 1.0749683380126953, -1.1670085191726685, 1.4645984172821045, 1.727752923965454, -0.3973279893398285, 0.15386104583740234, 0.376480370759964, 0.4587344527244568, -0.2894262671470642, 1.7622228860855103, 0.9494986534118652, 0.3337271809577942, -0.0004371096729300916, -2.8191640377044678, -1.2983442544937134, 2.6157851219177246, 1.2838857173919678, -1.6378134489059448, 0.673926055431366, 2.645265817642212, -0.9518860578536987, 0.597135603427887, -2.4177932739257812, 0.7742215991020203, 2.8373031616210938, -1.203477382659912, -0.34654930233955383, 0.5280588865280151, 0.3094676434993744, -0.9969009160995483, -1.3660107851028442, 0.4096437394618988, -3.8801841735839844, -1.9764256477355957, -0.923719048500061, 2.0230305194854736, 0.6894095540046692, -0.5597057938575745, -1.3345961570739746, -0.4470520317554474, 0.30333763360977173, 0.1739068627357483, -1.2200862169265747, 1.4873849153518677, -1.8531180620193481, -0.06866157799959183, -1.3397890329360962, 2.518282175064087, -2.778665781021118, 0.6305087804794312, -1.5617015361785889, -0.6771600246429443, 0.17890529334545135, 2.3487446308135986, -1.3164079189300537, -0.6421148180961609, -0.72955322265625, -3.8310317993164062, -1.0088573694229126, 1.5231199264526367, 1.2470849752426147, 1.290696144104004, 1.857024908065796, -0.00462189270183444, -0.3497069180011749, 0.17579376697540283, -1.444780707359314, -0.38028067350387573, -1.6545897722244263, -1.0689696073532104, -0.21408678591251373, -1.955392837524414, 0.4214446544647217, -0.08275127410888672, 1.1304630041122437, 4.833584785461426, 0.7460695505142212, -0.15022657811641693, -1.9767817258834839, -1.76903235912323, -1.069571614265442, 0.06376675516366959, -1.7948524951934814, -1.182967185974121, 2.604149341583252, -0.6337635517120361, 1.4784480333328247, 0.8402597904205322, 1.1146432161331177, -2.660635232925415, 1.009426236152649, 0.1809229850769043, -1.4721777439117432, -1.3682208061218262, -0.48750558495521545, 0.7318224310874939, -0.03963916748762131, -3.7949728965759277, 0.02702474407851696, 0.32178667187690735, -1.432945966720581, 0.03663339838385582, -1.2811981439590454, -1.5741424560546875, -0.055981628596782684, -0.5475453734397888, -0.701217532157898, 0.2606799900531769, 1.868246078491211, 0.9381178021430969, -0.5790606141090393, -0.07139991968870163, 1.6460107564926147, 1.0517098903656006, 0.704951286315918, 1.7030153274536133, 1.3921324014663696, -0.7834921479225159, -1.3335367441177368, 0.8225443363189697, -1.0667682886123657, 0.6180969476699829, 0.8686029314994812, -2.3131394386291504, 0.17892783880233765, -1.8471415042877197, -2.4270193576812744, -0.9085198640823364, 0.7317298650741577, -1.5044612884521484, 1.2398228645324707, 1.9311916828155518, 1.3107295036315918, -1.7828632593154907, -1.3346205949783325, -2.1450343132019043, -1.506840467453003, -0.25140243768692017], "sim": 0.3759448528289795, "freq": 1133.0, "x_tsne": 11.32407283782959, "y_tsne": 34.888389587402344}, {"lex": "bacteriophages", "type": "nb", "subreddit": "Coronavirus", "vec": [0.2941051423549652, 0.03740030527114868, -0.09898435324430466, -0.34987661242485046, 0.04828016832470894, 0.4723465144634247, -0.12387283891439438, -0.013993225060403347, -0.10084830224514008, -0.17671865224838257, 0.23845870792865753, -0.23210643231868744, 0.6617388129234314, -0.15836334228515625, 0.30396923422813416, -0.4341057240962982, -0.05486931651830673, -0.2089349329471588, 0.40248769521713257, 0.0681726410984993, -0.5954949855804443, 0.2179788053035736, 0.5638726949691772, 0.11393801122903824, 0.6634564399719238, -0.2884141206741333, -0.22980418801307678, 0.022686582058668137, -0.3521668612957001, -0.4999193251132965, -0.2912983298301697, 0.3668075501918793, -0.08574704825878143, 0.08792638033628464, 0.48635634779930115, 0.1811431348323822, 0.18575049936771393, -0.0947439968585968, -0.38052359223365784, -0.3591514527797699, -0.39075353741645813, -0.6554174423217773, -0.01347909402102232, 0.5339109301567078, -0.26284098625183105, -0.32542291283607483, 0.07708179950714111, 0.3448837697505951, -0.17806261777877808, -0.11288720369338989, -0.03992811590433121, -0.3386581540107727, -0.277398943901062, -0.175578773021698, 0.13702823221683502, -0.03140953183174133, -0.2152302861213684, 0.31948888301849365, -0.10817553102970123, -0.8665835857391357, 0.6864200234413147, 0.02413613721728325, 0.18890972435474396, 0.08504331856966019, -0.009322063997387886, -0.010195408016443253, -0.3747826814651489, 0.07219086587429047, -0.4146546721458435, 0.05809718742966652, 0.07554814219474792, -0.057982660830020905, 0.43576177954673767, 0.5121886134147644, -0.00015392436762340367, -0.45042696595191956, -0.4803725481033325, -0.20783556997776031, -0.28157156705856323, -0.14877480268478394, 0.11135941743850708, -0.26678967475891113, -0.03569233417510986, 0.11667683720588684, -0.14102493226528168, 0.21127139031887054, -0.41636255383491516, -0.2518075108528137, 0.049672044813632965, 0.8845856785774231, 0.10989008843898773, -0.25800713896751404, 0.21856288611888885, -0.009877856820821762, -0.10345996916294098, 0.08645881712436676, 0.2389007955789566, 0.4796707332134247, 0.5603682994842529, 0.18438945710659027, -0.2490355223417282, 0.3295426368713379, 0.14325404167175293, 0.1975165456533432, 0.13687953352928162, -0.029206691309809685, -0.006067351438105106, 0.3474312424659729, 0.0510326586663723, -0.03203711286187172, -0.1686246246099472, 0.22542080283164978, -0.013178940862417221, -0.3035357892513275, -0.2400670200586319, 0.20613044500350952, -0.08687072992324829, -0.2481110692024231, -0.20728598535060883, -0.008950560353696346, -0.3029833734035492, 0.23624959588050842, -0.1037716343998909, 0.024766236543655396, 0.44310757517814636, 0.42338427901268005, 0.1333712488412857, 0.3191814720630646, 0.4625358283519745, 0.49379920959472656, 0.10290056467056274, 0.4040542244911194, -0.060524195432662964, -0.45740267634391785, 0.23989485204219818, 0.03975804150104523, -0.1519673466682434, -0.15412144362926483, 0.11325737088918686, -0.49309590458869934, 0.3462922275066376, -0.1599450260400772, -0.053355105221271515, -0.16144506633281708, -0.13741913437843323, -0.42869696021080017, -0.14425967633724213, -0.051535580307245255, -0.2709280550479889, 0.2606388032436371, 0.23661905527114868, -0.28978925943374634, 0.13575083017349243, -0.09130410850048065, 0.23737826943397522, 0.055192939937114716, 0.2021668404340744, -0.24030660092830658, -0.48910874128341675, 0.010679259896278381, 0.182821586728096, -0.1432497352361679, -0.22547580301761627, -0.26309096813201904, -0.06330089271068573, -0.0679839551448822, 0.025736216455698013, 0.31708043813705444, 0.5779732465744019, 0.25490477681159973, 0.08210549503564835, 0.0814879760146141, -0.23685763776302338, 0.5147216320037842, 0.017109539359807968, -0.410049170255661, -0.3090222477912903, 0.3512505888938904, 0.06515099108219147, -0.15479616820812225, 0.041077882051467896, 0.13260211050510406, -0.09007430821657181, 0.03216353803873062, -0.009886241517961025, -0.2650723457336426, 0.43651965260505676, 0.10837160050868988, 0.10475750267505646, -0.0650562047958374, -0.12756168842315674, -0.227214977145195, -0.6924076676368713, -0.3022106885910034, 0.03530978783965111, 0.04103740304708481, 0.020813344046473503, 0.01999441720545292, 0.24903222918510437, 0.1586480438709259, -0.4294423460960388, 0.07518266886472702, -0.27191856503486633, 0.2901585102081299, 0.16102761030197144, -0.1832469403743744, -0.34362679719924927, 0.4694584608078003, -0.05524354800581932, -0.004635462071746588, -0.42367473244667053, -0.11546724289655685, -0.12077203392982483, 0.05711432546377182, 0.4709623157978058, 0.23667486011981964, -0.21644702553749084, -0.5285931825637817, -0.03609572723507881, -0.5010470747947693, -0.33431705832481384, 0.5241276621818542, 0.2207287698984146, 0.009186089038848877, -0.2551262676715851, 0.030243532732129097, 0.20484095811843872, -0.005002664402127266, -0.40662723779678345, -0.3858761489391327, 0.1250348538160324, -0.004143472760915756, -0.4475712776184082, 0.32478341460227966, 0.25401806831359863, -0.10361573100090027, 0.33890536427497864, 0.6794719099998474, 0.11007260531187057, 0.0156033830717206, -0.18245108425617218, -0.3557017743587494, -0.2170860469341278, -0.08160459250211716, -0.08292657881975174, -0.7947563529014587, 0.05497840419411659, 0.4075094163417816, -0.052317675203084946, 0.2443828135728836, -0.10518380254507065, -0.5681264996528625, 0.37503206729888916, 0.14945068955421448, -0.25348660349845886, -0.4264596104621887, -0.21746118366718292, 0.09439265727996826, 0.3323522210121155, -0.1976533681154251, 0.11808472126722336, -0.029819441959261894, -0.36735641956329346, -0.3529298007488251, 0.0709095224738121, -0.08798036724328995, 0.3690095841884613, 0.15553145110607147, 0.16588667035102844, 0.084108367562294, -0.10566650331020355, -0.2526334226131439, -0.510069727897644, 0.22619383037090302, 0.25023123621940613, 0.5344467163085938, 0.03559881076216698, 0.5832756161689758, -0.08738625794649124, -0.2700139582157135, 0.07714390754699707, -0.22768090665340424, -0.4360498785972595, 0.1459096372127533, -0.05374828353524208, 0.008639899082481861, -0.11493571847677231, -0.4441232979297638, -0.2731513977050781, 0.2504374384880066, 0.21563304960727692, 0.12004923820495605, 0.3853929936885834, 0.4245184361934662, 0.3064495325088501, -0.04606194049119949, -0.061446964740753174, 0.5357332229614258, -0.2760692834854126, -0.2841372787952423], "sim": 0.37538281083106995, "freq": 22.0, "x_tsne": 1.6091139316558838, "y_tsne": 4.613873481750488}, {"lex": "vx", "type": "nb", "subreddit": "conspiracy", "vec": [0.1737753301858902, 0.429370254278183, 0.5432335734367371, 0.17236849665641785, 0.04063479229807854, -0.7644172310829163, -0.3875078856945038, -0.03928975760936737, 1.1833022832870483, 1.429417371749878, 0.5830224752426147, -0.3283323049545288, 1.4214305877685547, -0.66838538646698, 1.3366210460662842, 0.7984417080879211, 0.1855761855840683, -0.09014169871807098, 0.35488516092300415, 1.4608700275421143, 0.506938636302948, 0.03562288358807564, -0.17139358818531036, -0.15948395431041718, 0.6943578720092773, -0.03352653607726097, 0.17785602807998657, 0.2616446018218994, 0.16516706347465515, 0.478287935256958, 0.7375442981719971, -0.02865111082792282, 0.660407543182373, -1.5236791372299194, 0.5346971154212952, 0.5115626454353333, -0.6326969265937805, 0.12883368134498596, -1.265039086341858, 0.7312111854553223, -0.3909708261489868, 1.1189606189727783, -0.6224691867828369, 0.2931213080883026, 0.5164637565612793, -0.7774675488471985, 1.3425332307815552, 0.3384951949119568, 0.5950390100479126, -0.29810208082199097, -0.10317053645849228, -2.2314980030059814, -0.20500722527503967, -0.8302897810935974, -0.8735037446022034, -0.4308604300022125, 0.051792923361063004, 0.5612481832504272, -1.0249724388122559, -1.7311534881591797, 0.1607843041419983, 1.4355825185775757, -0.41490209102630615, 1.0475249290466309, -0.24528014659881592, -0.3631765842437744, -0.7711793780326843, 0.18821728229522705, -0.18330520391464233, -0.4370604455471039, 1.356628179550171, -0.2550808787345886, -0.27327239513397217, 0.20353753864765167, -1.5769734382629395, -1.8759957551956177, 0.22426104545593262, 0.15689480304718018, -0.994782567024231, -0.08361086994409561, 0.3234148919582367, 0.4395400285720825, -0.48420798778533936, 1.030289649963379, -0.9924143552780151, 0.3500312566757202, -0.7670871615409851, -0.5987106561660767, -0.6241993308067322, 0.6180029511451721, 0.4511508047580719, 0.7468974590301514, 1.574556589126587, -0.183141827583313, -0.6284563541412354, -0.009250327944755554, 0.24534055590629578, -0.1900290846824646, 0.2910134792327881, 0.4509339928627014, -0.21454602479934692, -0.392104834318161, 0.19072626531124115, -0.37700769305229187, -0.6347103714942932, -1.0251693725585938, 1.3655065298080444, -0.49784162640571594, -0.7103356719017029, 0.3628597557544708, -0.05678778514266014, -1.1374324560165405, -1.2918987274169922, -0.35963869094848633, -0.612180769443512, 0.11492004245519638, 0.08037841320037842, -0.3394724428653717, 0.3434893786907196, -0.0767689049243927, 0.5431705117225647, -0.1536078155040741, -0.4511750638484955, 0.24370922148227692, -0.4114306569099426, 1.1823515892028809, 0.2192729413509369, -0.7344362735748291, 0.2160709947347641, 0.45795536041259766, -0.25098171830177307, 0.08805552870035172, 0.4579892158508301, 0.3425740599632263, 0.41607674956321716, -1.2523741722106934, -0.380954384803772, -0.7760748267173767, 0.4631319046020508, 0.7702268362045288, -0.8378557562828064, -0.8291687369346619, -0.4351729452610016, 1.3062024116516113, 1.399472713470459, -1.1044995784759521, -0.3319360017776489, -0.33041828870773315, 0.41718119382858276, -1.042432427406311, 0.6051623821258545, 0.5176799893379211, -1.7987866401672363, 0.12978790700435638, -0.13956353068351746, 0.6116248369216919, -0.9137815833091736, 0.22347183525562286, -0.5025389194488525, 1.486405849456787, -0.4205062687397003, 0.964691698551178, -0.09472338855266571, 0.5661672949790955, 0.16766932606697083, 0.045946381986141205, 0.46983954310417175, 0.7825399041175842, -0.3765333294868469, 1.0751471519470215, -0.10881005227565765, -1.62749183177948, -1.1206955909729004, 0.061764299869537354, -0.5375934839248657, -0.07425129413604736, -0.3288143277168274, -0.5644903779029846, -0.5963727235794067, 0.19297434389591217, -0.5587779879570007, 0.2691008448600769, 1.0223019123077393, 0.013778109103441238, 0.8456797003746033, 0.34388288855552673, 0.9965347647666931, 0.22908887267112732, 0.6979777812957764, 0.445905476808548, -0.4896426200866699, -0.0325983501970768, -0.3475973606109619, 0.14521607756614685, -0.28333693742752075, -0.31692296266555786, -0.7974310517311096, -0.2133878767490387, -0.15413475036621094, -0.32936158776283264, -0.42248305678367615, 1.298150897026062, -0.03207070007920265, 0.9143815040588379, -1.402725338935852, -0.2647880017757416, -0.42081090807914734, -0.4979917109012604, 0.46698564291000366, -0.568077564239502, -0.2776741683483124, 0.23491673171520233, -0.8335633277893066, 0.4777827560901642, -0.08866928517818451, 0.22451186180114746, -0.2845650911331177, -0.11973756551742554, -0.8239482641220093, -1.497550129890442, 1.1239808797836304, 0.46281859278678894, -0.2901611030101776, 0.0005005300045013428, -0.707875669002533, -0.5239247679710388, 0.009911984205245972, 0.8224676251411438, 0.39438727498054504, 0.6075414419174194, 0.022068556398153305, -0.8016504049301147, -0.21396543085575104, 0.4871058762073517, 1.2873116731643677, -0.5636293888092041, -0.7964637279510498, 1.3708817958831787, 0.286861777305603, 0.7171909213066101, 0.23205529153347015, -0.287360817193985, -0.43396520614624023, -0.862739622592926, 0.05403093248605728, 0.013128764927387238, 0.5591840744018555, 0.09693752229213715, 0.6751019358634949, 1.1934821605682373, -0.170278862118721, -1.618113398551941, 0.9953801035881042, 0.6826274394989014, 1.3065752983093262, 0.09827955067157745, -1.1305218935012817, -0.165741428732872, 0.8502088785171509, 0.04399518668651581, 0.6857635378837585, -0.5244141817092896, -0.9281063079833984, -1.3003733158111572, 0.35375508666038513, -0.2134549617767334, -0.06339029967784882, 0.5930711627006531, -0.5965104699134827, 0.7010462880134583, 0.993800699710846, 1.0232070684432983, -0.5799401998519897, -0.7285876870155334, 0.07201706618070602, 1.4636001586914062, 0.2072485089302063, 2.6083016395568848, 0.12965340912342072, -0.6057456135749817, -0.47695037722587585, -1.4920958280563354, 0.5556260943412781, 0.9665854573249817, -0.1409931480884552, -0.04096418619155884, 1.4260733127593994, -1.683899998664856, 0.003609136212617159, -0.30792587995529175, 0.007851220667362213, 0.489258348941803, -0.6407220959663391, 0.7713229060173035, 0.9479573369026184, 0.5643240213394165, -0.8493066430091858, 0.3418770432472229, -0.3981921076774597, -0.17148953676223755], "sim": 0.5895666480064392, "freq": 68.0, "x_tsne": -3.965569496154785, "y_tsne": -17.914892196655273}, {"lex": "neurotoxins", "type": "nb", "subreddit": "conspiracy", "vec": [0.5238630771636963, 0.15742288529872894, 0.28159019351005554, -0.6519985795021057, 0.4285832941532135, 0.7375240325927734, -0.22385463118553162, 0.6578503251075745, -0.2729094624519348, 0.5818069577217102, 0.0938107967376709, -0.8315087556838989, 0.5211623907089233, -0.9192084670066833, 0.37021809816360474, 0.11294174939393997, -0.30929750204086304, -0.46824246644973755, 0.35006046295166016, -0.20636112987995148, -0.5252994298934937, -0.05826696753501892, -0.23591375350952148, -0.04999132826924324, 0.7746545672416687, 0.016315467655658722, 0.5592769980430603, 0.3545246720314026, -0.2738940417766571, -0.009835131466388702, -0.36858654022216797, -0.3190722167491913, -0.23040080070495605, 0.15975886583328247, 0.11166985332965851, 0.5393723249435425, 0.5791982412338257, -0.1787482351064682, -0.9173542857170105, 0.17191335558891296, -0.5549734234809875, 0.10609559714794159, 0.24496442079544067, 0.07610547542572021, -0.2534959018230438, 0.2003755271434784, 0.6324365139007568, 1.004835844039917, 0.8998374938964844, -0.2770121097564697, 0.0435248464345932, 0.1423410177230835, -0.528511643409729, 0.8201319575309753, -0.7359788417816162, -0.10663650929927826, 0.15455889701843262, 0.19960704445838928, -0.041186340153217316, -1.4166620969772339, 0.147528737783432, -0.1087835431098938, -0.14719538390636444, -0.6701777577400208, 0.1931367814540863, 0.10200737416744232, -0.04576191306114197, -0.20935370028018951, 0.40604937076568604, 0.8298149704933167, 0.5964493155479431, 0.22419515252113342, -0.4690074324607849, 0.2558416724205017, 0.18554812669754028, -0.10929054021835327, -0.18732868134975433, -0.005528948735445738, -0.3276687264442444, -0.3320424258708954, 0.09195961803197861, 0.6673465967178345, -0.2879827320575714, 0.5830650329589844, -0.7923542261123657, -0.045799341052770615, -0.49336865544319153, -0.4363686740398407, 0.049898989498615265, -0.050010763108730316, -0.02546098083257675, -0.7251830101013184, 1.3226947784423828, 0.37242743372917175, -0.4506216049194336, 0.41177797317504883, 0.30677199363708496, -0.17970415949821472, 0.16457805037498474, -0.06286487728357315, 0.3584999740123749, -0.03602466359734535, -0.014339104294776917, -0.4999231994152069, -0.5335095524787903, -0.10713870823383331, -0.20596732199192047, -0.045983653515577316, 0.6147363781929016, 0.2630157768726349, -0.5951734781265259, -0.14478106796741486, -0.17354024946689606, 0.0688810721039772, -0.33977869153022766, 0.14914634823799133, -0.519760012626648, -0.21421588957309723, -0.3536650836467743, 0.07745084166526794, -0.29662981629371643, 0.6498948931694031, 0.3557884991168976, 0.4369108974933624, 0.2830202281475067, 0.7374866604804993, -0.07059456408023834, -0.7716372013092041, 0.19159364700317383, 0.34556251764297485, 0.37327075004577637, 0.34648290276527405, -1.1551674604415894, 0.5541409254074097, -0.24197137355804443, -0.5184357166290283, -0.49750593304634094, -1.2404143810272217, 0.3574363589286804, 0.20490220189094543, 1.0872350931167603, -0.16171391308307648, -0.1956324726343155, 0.043665505945682526, -0.5513802766799927, -0.5440379977226257, -0.6337795257568359, -1.3380337953567505, 0.1720222681760788, -0.35149770975112915, 0.31932690739631653, 0.22854045033454895, -0.11817815899848938, -0.05578786879777908, -0.6536027193069458, 0.03648979216814041, -0.8463798761367798, 0.2930282950401306, -0.7386865615844727, 0.2820950746536255, -0.11511743813753128, 0.7791917324066162, -0.2567797303199768, -0.09192748367786407, -0.10100644081830978, -0.1825711578130722, -0.7808891534805298, 0.25993236899375916, -0.04484715312719345, 0.2890661954879761, -0.06957246363162994, 0.6033110022544861, -0.4056210219860077, 0.11250324547290802, 0.2367987036705017, 0.2741692066192627, -0.31238067150115967, 0.6438423991203308, -0.10509016364812851, 0.2849594056606293, -0.02717277780175209, 0.3585304915904999, 0.24167074263095856, -0.011609368026256561, 0.03246389701962471, -0.01127217523753643, 0.6401302218437195, 0.056716978549957275, 0.8823133111000061, 0.03858399763703346, -0.020550094544887543, -0.12844114005565643, 0.49527034163475037, 0.31762146949768066, 0.009679041802883148, -0.36432451009750366, -0.23563408851623535, -0.18424387276172638, 0.6848336458206177, 0.35520607233047485, -0.6153713464736938, 0.8012911081314087, 0.3361701965332031, 0.5390248894691467, -0.36788368225097656, -0.45626717805862427, -0.3742920756340027, -0.0064995624125003815, -0.5621435046195984, -0.01281682401895523, -0.5998624563217163, -0.2208951711654663, -1.044290542602539, 0.04277355968952179, -0.11650953441858292, -0.18526768684387207, -0.4917930066585541, -0.2159401774406433, -0.37107259035110474, -0.7267438173294067, 0.05911403149366379, 0.24374087154865265, -0.03696206212043762, -0.18633897602558136, -0.46063873171806335, 0.301177442073822, 0.24956899881362915, 0.7516752481460571, 0.13449010252952576, 0.3633805215358734, 0.3069142997264862, -0.029709838330745697, -0.24111953377723694, 0.5153146982192993, 0.6279537081718445, -0.9061056971549988, 0.47217264771461487, 0.5151342749595642, 0.5202491879463196, 0.10318337380886078, -0.4325036406517029, -0.01110195554792881, 0.07428580522537231, -1.0180394649505615, -0.11846134066581726, -0.715898871421814, 0.43377238512039185, 0.478516548871994, 0.022194676101207733, 0.2998136878013611, 0.2298395186662674, -0.7980974912643433, 0.5105710029602051, 0.3924870789051056, 0.7330453991889954, 0.1722465455532074, -0.530524730682373, -0.2589357793331146, -0.7377109527587891, 0.17649216949939728, 0.31355226039886475, -0.6441215872764587, 0.21131755411624908, -0.6628873944282532, 0.40498387813568115, 0.09355008602142334, 0.07599733769893646, 0.2098640352487564, -0.20117606222629547, -0.1783604770898819, -0.2771730422973633, -0.3616580665111542, -0.6843135356903076, -0.6450636982917786, 0.5196219086647034, 0.2330026626586914, -0.5512639880180359, 0.8148321509361267, 0.09416218101978302, -0.09348052740097046, -0.6748709678649902, 0.3869943916797638, -0.183386892080307, 0.07232832908630371, 0.9124467372894287, 0.7890124320983887, 0.270297110080719, -0.8009517192840576, -0.8097824454307556, -0.3217446804046631, 0.06925687193870544, 0.062487028539180756, 0.19325196743011475, 0.13266949355602264, 1.0122102499008179, 0.6496407985687256, -0.3999992907047272, 0.6341596841812134, -0.48627033829689026, 0.04708648473024368], "sim": 0.517458975315094, "freq": 33.0, "x_tsne": -18.81086540222168, "y_tsne": -1.5216259956359863}, {"lex": "gmos", "type": "nb", "subreddit": "conspiracy", "vec": [0.5870979428291321, -0.22144931554794312, -0.5366806983947754, -1.6395111083984375, 0.33222681283950806, 0.4674198627471924, 0.07677234709262848, -0.3374069035053253, -0.48384514451026917, -0.16720545291900635, -0.07044342160224915, -1.0266568660736084, 1.1849015951156616, -0.31171396374702454, 2.095391273498535, 0.7849085330963135, 0.23985496163368225, -0.9225134253501892, -1.2731982469558716, 0.3192403018474579, -1.1564152240753174, -0.4531964361667633, -0.5077261328697205, 1.649385929107666, 1.6057515144348145, 0.6594502925872803, -0.5720164775848389, -1.2620538473129272, -0.03718996420502663, 0.7407972812652588, -0.07158657908439636, -0.21683591604232788, 1.3491103649139404, -1.840456247329712, 0.007288267835974693, 0.46370449662208557, -0.2619251608848572, -0.4217020273208618, -1.7917135953903198, 1.009832739830017, -2.065070390701294, 1.1478140354156494, -0.7436623573303223, 0.7310119271278381, -0.06184593588113785, -0.24050962924957275, 1.4907634258270264, 2.338498592376709, 1.6058186292648315, -1.4070760011672974, 0.033823296427726746, -0.3383505046367645, -0.13852402567863464, 1.24544095993042, -0.8839306235313416, 0.6744934916496277, 1.3026020526885986, 0.6413508057594299, -0.03729791194200516, -2.588449716567993, 1.4225772619247437, 0.23664847016334534, -0.5002270936965942, 0.7921189665794373, 0.2747761607170105, 0.8631755113601685, 0.20161087810993195, -0.9488750696182251, 1.631748914718628, 1.4210193157196045, 1.682850956916809, -1.4211680889129639, -0.33704355359077454, -0.8432259559631348, 0.8447788953781128, -2.381094217300415, 0.07788163423538208, 0.8810601234436035, -0.004526423290371895, 0.02797650545835495, 0.9290896058082581, -0.8734952807426453, -0.31323230266571045, -0.8240506052970886, -1.4662895202636719, -0.5819392204284668, 0.030383586883544922, -1.2402398586273193, -0.6881656050682068, -0.07040577381849289, -0.47324392199516296, -0.7469097375869751, 1.9780749082565308, 0.8200696706771851, -0.37974417209625244, 0.4457187354564667, 1.174127459526062, -0.4448089003562927, 0.39887765049934387, 1.258863091468811, 0.6458492875099182, -1.250580906867981, -0.11879812926054001, -0.6780339479446411, 0.178533136844635, -0.5586288571357727, 0.29216668009757996, -1.0834145545959473, -0.09434163570404053, -0.33575230836868286, -2.36026930809021, -2.075002670288086, 0.16197791695594788, 0.4933946430683136, 0.49461373686790466, -0.1434115171432495, -0.44388261437416077, -1.7539066076278687, 0.8859617710113525, 0.6193069219589233, 1.303126335144043, 2.1416516304016113, -0.22435501217842102, -0.15440815687179565, -0.4754120111465454, 0.9561346173286438, -0.0026869475841522217, -1.6120491027832031, 0.449390172958374, 0.7226250171661377, -0.34587937593460083, 1.0472924709320068, -1.460207223892212, 0.03245055675506592, -0.529360830783844, 0.05266040563583374, -1.6593986749649048, -1.8221800327301025, 0.6723985075950623, -0.20290741324424744, 1.8181631565093994, -1.5454407930374146, 1.2083721160888672, -0.220631405711174, -0.22666198015213013, -1.3030905723571777, -0.12216690927743912, -0.3365064561367035, -0.5655757188796997, -1.4729024171829224, 0.9074985980987549, -0.7322365045547485, 0.02428285963833332, -1.8183008432388306, 0.18537789583206177, -0.11807713657617569, -1.3857133388519287, -0.9739913940429688, -1.080626368522644, -0.012786181643605232, 1.1098514795303345, -1.1122294664382935, 0.050983063876628876, -1.3795063495635986, -0.10667812824249268, 0.8327867388725281, -1.0599336624145508, 1.5781949758529663, -1.1063387393951416, 0.7388326525688171, 1.209733009338379, 0.8672093152999878, -0.32033953070640564, -1.2536864280700684, 0.9536909461021423, 0.039954282343387604, 0.17916925251483917, 1.459760308265686, 0.9705814719200134, -0.52972811460495, 1.4004336595535278, 2.2753360271453857, -0.3641853630542755, 0.6865030527114868, 1.3662137985229492, -1.3698813915252686, 2.5213615894317627, 1.1222225427627563, 0.10293900221586227, -0.10362880676984787, -0.4651018977165222, -0.006174897775053978, -1.3187960386276245, 0.6142156720161438, 0.2945408225059509, 0.02197262831032276, -0.7680535316467285, -0.08165065944194794, 1.7796096801757812, 0.9168566465377808, -0.6232554316520691, -0.426445871591568, -0.853907585144043, 0.8370790481567383, -0.906886875629425, -0.6926862597465515, -0.8084889054298401, 0.19949117302894592, 0.3426828980445862, -0.02466730959713459, -2.446812391281128, -0.5508239269256592, -1.6153931617736816, -0.5283829569816589, 0.9765983819961548, -0.869064211845398, -1.0096714496612549, -0.6813206672668457, -0.14005036652088165, 0.22828753292560577, -1.2727781534194946, 1.4401893615722656, 0.5941942930221558, -0.6672473549842834, -0.5755756497383118, 0.19482238590717316, 0.8162702322006226, 0.5152223706245422, 0.5983223915100098, 0.3192870020866394, 0.12207328528165817, 0.17492830753326416, 0.11069509387016296, 0.39098113775253296, 2.276693105697632, -0.7031174302101135, -0.21645034849643707, 0.6002833247184753, 1.4019348621368408, -0.511361300945282, 0.31760090589523315, -0.4979444444179535, -0.6898967027664185, -0.9903197288513184, -0.18282289803028107, 0.6027416586875916, 0.3791019916534424, -0.5635082721710205, 0.7535244822502136, -0.2966306507587433, -0.2927848696708679, -0.9126738905906677, 0.9157360196113586, 0.08610451966524124, 1.1677912473678589, 0.20268341898918152, -1.6816543340682983, -1.022449016571045, -1.3367350101470947, -0.6425899267196655, 1.6565395593643188, 0.3664325773715973, 0.2902062237262726, -1.3366681337356567, -0.7437629699707031, 0.6230770945549011, 1.0161579847335815, 0.6127879619598389, 1.2786533832550049, -0.3934486508369446, -1.0672396421432495, -1.1867268085479736, -1.259566068649292, -0.8622311353683472, 0.9583267569541931, -0.16778694093227386, -0.0754338726401329, 0.16759061813354492, -0.0675790086388588, -0.6614874005317688, -0.6023836135864258, 0.25830695033073425, 0.4868912994861603, -0.327799916267395, 0.8911021947860718, 0.1614067256450653, 0.8053538799285889, -0.6390159130096436, 0.015492476522922516, 0.7926167845726013, -0.662007212638855, -1.8093620538711548, 1.5034040212631226, 0.15519706904888153, 0.6695282459259033, 0.6844716668128967, -0.24299226701259613, 1.4547486305236816, -0.34651121497154236, 0.8861128687858582], "sim": 0.5110951066017151, "freq": 179.0, "x_tsne": -25.4161434173584, "y_tsne": -4.067355155944824}, {"lex": "pesticides", "type": "nb", "subreddit": "conspiracy", "vec": [0.6999111771583557, 0.1413099318742752, -0.7428778409957886, -1.9287748336791992, 0.19069397449493408, 1.445552945137024, 0.7150343656539917, 2.5148637294769287, -0.5592272877693176, 0.018204856663942337, -0.04909240081906319, -1.557982325553894, 1.701273798942566, -0.742588222026825, 0.8408153057098389, -0.49323567748069763, 0.6942400336265564, -0.7890410423278809, -0.3301708698272705, -1.678780198097229, 0.2891273498535156, 0.6070172190666199, -0.29029518365859985, 0.15512000024318695, 3.5438497066497803, -0.11740720272064209, 0.6635100841522217, -0.00662365835160017, -0.693365752696991, -0.28245043754577637, 0.18662023544311523, 0.15498630702495575, 0.5601944923400879, -1.313219666481018, 0.4848712682723999, 2.609744071960449, 0.17300990223884583, 0.5035105347633362, -2.3508853912353516, 0.055903851985931396, -2.2964253425598145, -0.09168460965156555, -1.8442498445510864, -0.03269956633448601, -0.18207041919231415, -0.18071721494197845, 3.264472723007202, 1.2877960205078125, 3.306135654449463, -1.6872785091400146, -0.01650170236825943, -1.4247907400131226, 0.3299846053123474, 1.5505973100662231, -2.25166392326355, 0.988440215587616, 0.8645301461219788, -0.058150798082351685, 0.6147719621658325, -1.786291241645813, -0.9300556778907776, 1.2143126726150513, -0.23754775524139404, -0.7423989772796631, 1.623464822769165, 0.15596498548984528, 1.7025976181030273, 0.4516046345233917, 1.8382890224456787, 1.2421340942382812, 0.4061641991138458, -0.6199232935905457, -0.22651657462120056, -0.2965867817401886, 0.2950791120529175, -1.6401652097702026, -0.7047728896141052, 0.9788244366645813, 0.04262903332710266, -0.6667312383651733, 0.7131246328353882, -1.0015758275985718, -1.1795125007629395, 0.4186840057373047, -2.087437868118286, 0.4667912423610687, -1.0480669736862183, -0.37905153632164, 0.17888996005058289, -0.30796128511428833, -0.019010325893759727, -2.119391918182373, 1.3962547779083252, 1.1037883758544922, -2.0821638107299805, 0.027800846844911575, 1.0558112859725952, -0.7208210229873657, -0.1852579414844513, 1.695488452911377, 1.364590048789978, 0.7556333541870117, -0.4422007203102112, -0.6059350967407227, -1.3360711336135864, -1.1149495840072632, -0.4598214328289032, 0.5664097666740417, -0.6442419290542603, 0.7439822554588318, -1.4659794569015503, -0.8692636489868164, -1.2572931051254272, -0.7323424220085144, -0.2411414384841919, -0.5984767079353333, -0.4861987233161926, -1.8174611330032349, -0.795219361782074, 1.4794718027114868, 1.5216039419174194, 2.10489559173584, -0.02531224675476551, 0.66233891248703, 1.9436683654785156, 2.928266763687134, 1.1602622270584106, -2.205824136734009, 1.9208905696868896, 1.664729118347168, -2.155578851699829, 2.600663661956787, -1.510284662246704, 0.1287490725517273, -0.3679717481136322, 0.293381005525589, -0.1412438154220581, -2.3499741554260254, 1.4559751749038696, -0.066628098487854, 1.5536237955093384, -1.515127182006836, -0.25595584511756897, 0.49087437987327576, -0.4746692180633545, -3.039478063583374, 0.1513257771730423, -0.8401393294334412, -0.6779488325119019, -1.0875506401062012, 0.04030068591237068, -1.75275719165802, -0.4093402028083801, -1.651808738708496, -0.48428410291671753, -0.1459398865699768, -2.2325470447540283, -1.6542795896530151, -0.4469214081764221, 0.514835774898529, 2.1994552612304688, -1.3504142761230469, -1.6330944299697876, -1.1828066110610962, 1.1636332273483276, 0.7187976837158203, -0.25127366185188293, 0.448733389377594, -0.8354822397232056, 0.7288610935211182, 2.5761680603027344, 0.44015610218048096, -2.1585276126861572, 0.3630671203136444, 0.30299434065818787, 0.3866102993488312, -1.3954638242721558, 0.20280461013317108, 1.0162031650543213, -1.106034755706787, 1.0972665548324585, 2.70343017578125, 0.6628613471984863, 0.6325269937515259, -0.05121973901987076, -0.5656761527061462, 2.06917142868042, 0.15739059448242188, 1.2554222345352173, -0.9744831323623657, -0.7613896727561951, -1.003193974494934, -0.10004271566867828, 1.5198277235031128, -0.5060129165649414, -0.24682100117206573, -0.016061197966337204, -0.032444972544908524, 1.0052725076675415, 0.6576531529426575, -0.7267635464668274, 0.19977128505706787, 0.42126885056495667, 0.17229856550693512, -1.8966234922409058, -0.14050807058811188, -0.9222597479820251, 0.565405011177063, 0.21369527280330658, 0.628216564655304, -3.0708582401275635, 0.1648963987827301, -0.6514983177185059, -0.5224527716636658, 0.06367653608322144, -1.0070102214813232, -0.7064332365989685, -0.910088062286377, 0.9757659435272217, -0.2325144112110138, 0.9826972484588623, 2.0087971687316895, -0.8428341746330261, -1.357197642326355, -0.22129815816879272, 0.9088821411132812, 1.3933720588684082, -1.3205256462097168, 0.4422135651111603, -0.31519123911857605, 0.490000456571579, -1.161726474761963, -1.6909908056259155, 0.7896299958229065, 1.970326542854309, -1.170850396156311, 0.27320829033851624, 2.58734130859375, 1.6763712167739868, -1.0935027599334717, 0.04938282445073128, -0.06360192596912384, -1.4218376874923706, -0.5835280418395996, -1.8490334749221802, 0.23275592923164368, 2.692924737930298, 0.9052048921585083, 3.0180399417877197, 0.6041456460952759, -0.6386919021606445, -1.0921874046325684, 1.3477014303207397, 0.39814916253089905, 1.65369713306427, -1.3309086561203003, -2.4642765522003174, -1.7442551851272583, -0.7280588746070862, -0.0313384011387825, 1.036683440208435, 0.12220977991819382, 1.1436448097229004, -3.4360005855560303, -0.5522916316986084, 0.08601054549217224, 0.6544591784477234, 0.371294766664505, 2.296632766723633, -1.3625296354293823, 0.057486724108457565, -0.6186595559120178, 0.6567481756210327, -0.5595883727073669, 0.7678240537643433, 0.5517246127128601, 2.299041986465454, 1.2246284484863281, 0.06587100774049759, -0.930152416229248, -1.3274507522583008, 0.02063395082950592, -1.207302212715149, -0.4749777913093567, 1.1335370540618896, -0.43283748626708984, 1.7327899932861328, -0.022395726293325424, -1.4786306619644165, -0.4139653146266937, -1.1490848064422607, -0.435799241065979, 0.6644077301025391, 1.1730380058288574, 0.9695111513137817, 0.6356996297836304, 0.6815329194068909, 0.5696626305580139, -0.9228905439376831, -0.5987095236778259], "sim": 0.5029253959655762, "freq": 291.0, "x_tsne": -25.825937271118164, "y_tsne": -1.3719784021377563}, {"lex": "ipv", "type": "nb", "subreddit": "conspiracy", "vec": [-0.04805618152022362, -0.006864459253847599, 0.8576858639717102, -0.5539212226867676, 0.234422504901886, 0.19790703058242798, 0.11825164407491684, -0.26827165484428406, 0.7326672077178955, 0.7520674467086792, 0.2111324518918991, -0.8358967304229736, 1.1634756326675415, -0.6469183564186096, 0.2192220687866211, 0.606559157371521, -0.9847705364227295, -0.5040400624275208, 0.11194700747728348, 0.29133889079093933, -0.4438652992248535, 0.25266286730766296, -0.322252482175827, 0.5918275713920593, 0.9147268533706665, -0.31747180223464966, 0.2626042664051056, -0.4493046998977661, -0.3947732448577881, -0.1019444540143013, -0.17193284630775452, -0.9721292853355408, 0.24732325971126556, -0.049464475363492966, -0.036018963903188705, 0.350809782743454, 0.2258928418159485, -0.17352581024169922, -0.665382444858551, 0.5882542133331299, -1.0211764574050903, 0.7372530698776245, 0.14363409578800201, -0.07489389926195145, 0.5866488218307495, 0.4570949077606201, 0.9937357902526855, 0.431135356426239, 0.08404120802879333, 0.6794339418411255, -0.0032409485429525375, -1.277745246887207, -0.08795468509197235, 0.6907393932342529, -0.2546241283416748, -0.655600368976593, -0.5465474128723145, 0.6028990745544434, 0.15386143326759338, -1.0192445516586304, -0.24072667956352234, -0.09336701035499573, 0.17099948227405548, 0.6536917090415955, 0.36284855008125305, 0.6529961228370667, 0.17097818851470947, 0.5519413948059082, 0.4054926931858063, 1.155701994895935, 0.7137143015861511, -0.4852147102355957, -0.43771305680274963, -0.44904905557632446, -0.07424158602952957, -0.17526917159557343, 0.2612440884113312, -0.523032009601593, -0.40834277868270874, -0.6884772777557373, 0.6866354942321777, -0.06322787702083588, -0.9016604423522949, -0.33514222502708435, -0.6408193111419678, 0.5964113473892212, 0.046234216541051865, -0.6882572770118713, -0.09243010729551315, -0.128302663564682, 0.22145824134349823, -0.05922551825642586, 1.046427607536316, -0.37271714210510254, -0.40754443407058716, -0.5653576850891113, -0.11596386879682541, -0.24237003922462463, 0.5776962637901306, 0.2176097184419632, 0.6830553412437439, 0.42873746156692505, 0.22257713973522186, 0.5246527194976807, -0.8695706129074097, -0.09074345231056213, 0.05149221420288086, -8.107651956379414e-05, 0.2664647698402405, 0.9340623021125793, -1.0650280714035034, -0.6238924264907837, 0.14147482812404633, 0.051643554121255875, -0.29240351915359497, 0.01982855796813965, 0.32418572902679443, -0.6066311001777649, 0.1856052577495575, 0.3355564773082733, -0.2391069382429123, -0.10956066101789474, -0.19290658831596375, 0.7367557883262634, -0.7674723863601685, 1.2242677211761475, 0.03183639049530029, -0.0779084861278534, 0.176432803273201, 0.7537931203842163, 0.39530661702156067, 0.3949035406112671, -1.0025848150253296, -0.48338258266448975, 0.36214444041252136, -0.11753696203231812, -0.49787184596061707, -0.361239492893219, -0.6569885015487671, 0.45230603218078613, 0.4122074246406555, -0.5366562008857727, -0.3602053225040436, 0.6729958653450012, 0.5583814382553101, -0.8410754203796387, -0.2565487027168274, -0.33478599786758423, 0.6095542907714844, 0.33288803696632385, -0.0208927970379591, -0.8764866590499878, -1.3894015550613403, -0.3144027292728424, -0.2394026517868042, -1.205628752708435, -0.9283144474029541, -0.1764025241136551, -0.9853637218475342, 0.37560051679611206, 0.19841207563877106, 0.379627525806427, -0.8231921195983887, 0.811678946018219, -0.29569151997566223, -0.8845874071121216, -0.740875244140625, 1.0584162473678589, 0.8741434812545776, 0.7927438616752625, 0.16558726131916046, 0.07144223153591156, -0.37060075998306274, -0.7063031196594238, 0.2958035171031952, -0.8953608274459839, -0.47579726576805115, 0.08690407872200012, -0.22843341529369354, 0.10597936064004898, -0.6036965250968933, 0.9167189002037048, -0.5370725393295288, -0.14308810234069824, 0.11725062876939774, -1.7106786966323853, 0.24372300505638123, -0.8197339177131653, 0.3324638605117798, 0.06474145501852036, -0.4477946162223816, 0.28676363825798035, -0.737388551235199, -0.14534467458724976, 0.23504050076007843, 0.027365121990442276, 0.15829116106033325, 0.39544862508773804, 0.833977460861206, 0.28249531984329224, -0.7984389066696167, 0.8618468642234802, -0.3996882140636444, 0.799024760723114, 0.041572246700525284, -0.3767751157283783, -0.5711939334869385, -0.09284156560897827, 0.5427929759025574, 0.7610427737236023, 0.11008931696414948, -0.6403387784957886, -0.16502051055431366, 0.550279974937439, 0.44527700543403625, 0.14349713921546936, -0.41038572788238525, -0.6275771260261536, -0.2509145438671112, -0.9116003513336182, 0.44795137643814087, 0.4355003535747528, 0.8274403810501099, 0.002948217559605837, -0.27926889061927795, -0.1311907172203064, 0.017892278730869293, 0.6566450595855713, -0.1601291447877884, 0.05722973123192787, 1.0266072750091553, -1.069100022315979, 0.3689361810684204, 0.3112771511077881, 0.9974664449691772, -0.3229740858078003, -0.14081360399723053, 1.3816609382629395, 0.20690837502479553, -0.09470713883638382, -0.27204763889312744, 0.05577405169606209, 0.3636816143989563, -1.0121784210205078, -0.09530190378427505, 0.9810155630111694, 0.28485116362571716, -0.3911803662776947, -0.3632008731365204, 0.6620311737060547, 0.38765570521354675, -0.308838427066803, 0.9020206332206726, 0.25187575817108154, -0.7242510318756104, 0.8101037740707397, -0.555148720741272, -0.773951530456543, 0.6607293486595154, 0.1349991261959076, 0.42211028933525085, 0.32838839292526245, -0.690930187702179, -0.9902325868606567, -0.3191237449645996, 0.27249592542648315, -0.2567645311355591, -0.09449711441993713, -0.03898000717163086, -0.07885371893644333, -0.6767354607582092, -1.5677685737609863, -1.0043891668319702, 0.01595919393002987, 0.26564738154411316, -0.009814898483455181, -0.404845267534256, 1.241496205329895, -1.0331089496612549, -0.5797751545906067, -0.5688958764076233, -0.4650983512401581, -0.47880688309669495, -0.25879696011543274, 0.3144529461860657, 0.44459959864616394, 0.7732652425765991, -0.3872157037258148, -0.082761250436306, -0.4681837558746338, 0.8815337419509888, 0.7367584109306335, -0.10300178825855255, 0.5832349061965942, 0.7453253269195557, -0.24285128712654114, -0.9023271799087524, 0.16711170971393585, 0.03419606387615204, -0.7093938589096069], "sim": 0.48746630549430847, "freq": 43.0, "x_tsne": -1.633658766746521, "y_tsne": -22.901540756225586}, {"lex": "measles", "type": "nb", "subreddit": "conspiracy", "vec": [-0.48496562242507935, 0.4851441979408264, 3.5525519847869873, -1.0103716850280762, 0.690528929233551, -0.8887385129928589, 0.24439094960689545, -1.3454071283340454, 0.8886998891830444, 2.8527259826660156, 0.05756551772356033, -1.6666792631149292, -0.5263023972511292, -1.568277359008789, 0.0846685916185379, 0.9215268492698669, 1.2294254302978516, -0.3912631869316101, 0.13155557215213776, 0.6801341772079468, 1.2423908710479736, 0.2297632098197937, 0.8425524830818176, 0.06864115595817566, 2.4866437911987305, 0.83018559217453, -2.4032249450683594, 0.5784429311752319, -1.233146071434021, 0.5204087495803833, -1.3353832960128784, -3.2377471923828125, 2.076946496963501, -1.443954348564148, 0.2846951484680176, 0.6542466282844543, -1.403417944908142, -0.4965994358062744, -2.5062856674194336, 0.5263521075248718, -1.6395747661590576, 2.954259157180786, -1.2716104984283447, 1.210384726524353, 0.15946656465530396, 0.5448659062385559, 1.257987141609192, 0.05527956783771515, 1.3871355056762695, 0.3959466218948364, 1.6106773614883423, -2.2177584171295166, 0.6746547222137451, 0.8502129912376404, -0.6594250202178955, -2.322035312652588, 0.513060986995697, 0.6597440838813782, 1.3030381202697754, -0.9610887765884399, 1.0677452087402344, 2.300025701522827, 0.1137155145406723, 1.042527198791504, 0.9385420083999634, 0.13224107027053833, 1.0956387519836426, 1.3856561183929443, -0.19605165719985962, 2.108609437942505, 0.5401620268821716, -1.0408570766448975, -0.439907044172287, 1.2236665487289429, -3.261892557144165, -1.2761327028274536, -0.584968626499176, -1.043339490890503, -1.731101393699646, -1.3023936748504639, -0.33219605684280396, 1.2879053354263306, 0.05062393844127655, -0.84797602891922, -0.4100542366504669, 1.196987509727478, -1.8861596584320068, -3.0891828536987305, 0.6222608685493469, -0.41466301679611206, 1.0392619371414185, 0.8436307311058044, -0.7817423939704895, 0.9563908576965332, -0.0004372522234916687, -0.44008588790893555, 0.8273231983184814, 1.9536908864974976, 0.2755358815193176, 0.18621425330638885, 2.2752153873443604, -0.1787116825580597, 1.9248582124710083, 1.0202007293701172, -1.912597417831421, 0.6875225305557251, -1.0535366535186768, -0.6025851964950562, 1.8434529304504395, 0.4166608452796936, -0.4895305633544922, -2.035738229751587, -1.5871899127960205, -0.6266627907752991, -1.183087706565857, -0.8059263825416565, 0.2845120429992676, 0.5889338850975037, -0.6328020095825195, 0.9603455066680908, -2.9002180099487305, -0.4931544363498688, 0.9266536235809326, -0.25342997908592224, 0.48218834400177, 1.3152086734771729, 0.8092136383056641, 0.22285176813602448, -0.6973173022270203, 1.7425739765167236, -0.03768941015005112, 0.3107623755931854, 2.323364496231079, -1.110349178314209, 1.6130249500274658, 0.882793664932251, -1.6568613052368164, -1.2543593645095825, 0.5733435153961182, -0.7529731392860413, -0.1894822120666504, -2.062643051147461, -0.7542528510093689, 2.589688539505005, 0.7901536226272583, -3.0006906986236572, -0.3035163879394531, -1.6103389263153076, 0.4021373689174652, -0.4110315144062042, -0.18803969025611877, 1.7837973833084106, -2.825752019882202, -1.5652859210968018, -0.9271910190582275, -0.6761338710784912, -0.6406833529472351, 1.7206509113311768, -1.8951359987258911, 2.096651315689087, 0.12336362898349762, 1.4800773859024048, -1.0808104276657104, -0.18469467759132385, 0.4720432460308075, -2.889814615249634, -0.2336517870426178, 1.784313678741455, 1.5770409107208252, 1.570884346961975, 0.051570940762758255, -1.3697892427444458, -0.20650595426559448, -1.1055688858032227, -0.9319028258323669, 0.13968177139759064, -1.1055045127868652, 0.2967052757740021, -3.087602138519287, 0.03692621365189552, -1.847779631614685, 1.262561559677124, 0.50564044713974, -2.355790138244629, -1.0680145025253296, -1.559944748878479, 0.5601499080657959, -0.1862683892250061, -0.17608200013637543, 0.3400249779224396, 0.14935460686683655, -2.0201010704040527, 0.15375995635986328, 1.1643290519714355, 1.7915899753570557, 1.9976495504379272, -0.47767555713653564, -0.7196137309074402, -0.06509491056203842, 0.7300959229469299, -1.6464356184005737, 1.0932340621948242, -0.8550795316696167, 0.2584245204925537, -1.7737585306167603, -0.867996871471405, 0.4189906120300293, -0.3534516990184784, -0.42931655049324036, 0.3308138847351074, -0.5320948958396912, -1.474705457687378, 2.485647439956665, 2.5481953620910645, 0.30522993206977844, 0.5305718183517456, 0.059470079839229584, -1.8393536806106567, -1.289441704750061, 1.0875985622406006, 2.02225399017334, -1.791771411895752, -0.23471523821353912, 0.2520175874233246, -2.5843875408172607, 1.456393837928772, 0.22940215468406677, 0.7396681308746338, 0.1555875986814499, -0.45482078194618225, 2.6780948638916016, -0.5693370699882507, -0.07334683835506439, 1.742566704750061, 3.5013699531555176, 0.35302335023880005, 1.3151780366897583, 2.616572380065918, 1.2449675798416138, -1.0939357280731201, 1.7288039922714233, -0.07879989594221115, 1.9670161008834839, -0.8790497183799744, -2.0561718940734863, 0.14140790700912476, 3.8749358654022217, -2.028801679611206, 3.257169008255005, 0.3080829977989197, -1.5873464345932007, -1.839829683303833, 1.0628610849380493, 1.8357791900634766, 0.7447179555892944, 0.136624276638031, -0.4268655776977539, -0.5739461779594421, -1.9105113744735718, 1.3650037050247192, 1.5913480520248413, -0.6071123480796814, -1.2768183946609497, -2.3654603958129883, -0.8737643957138062, 0.13639311492443085, 0.36174094676971436, 0.7070913314819336, 1.2209036350250244, 2.13572096824646, -0.6552947163581848, -2.7356765270233154, -1.634783148765564, -0.1523905098438263, 0.018155626952648163, 1.486283779144287, -0.3060213029384613, 3.957930088043213, -1.6268514394760132, 0.6101482510566711, -2.3707387447357178, 1.1186548471450806, -2.6431872844696045, 0.028695181012153625, 0.7422163486480713, -1.9438097476959229, 1.8041081428527832, 0.505667507648468, -0.363545298576355, 0.15842415392398834, 1.2332584857940674, 2.022768497467041, 1.0081931352615356, 2.1443891525268555, 0.26377177238464355, 0.8099944591522217, -0.9339706897735596, 2.2789394855499268, -3.5710628032684326, -2.8458945751190186], "sim": 0.4671691358089447, "freq": 2704.0, "x_tsne": -8.65066909790039, "y_tsne": -28.046329498291016}, {"lex": "vacs", "type": "nb", "subreddit": "conspiracy", "vec": [-0.12179644405841827, -0.4751621186733246, 0.5053396224975586, 0.4167753756046295, -0.3351469337940216, -0.47032853960990906, -0.014174720272421837, 0.010112076997756958, -0.37725508213043213, 0.861119270324707, 0.2561732530593872, 0.029162896797060966, 0.33248934149742126, -0.09642557054758072, 0.6247714757919312, 0.6229261755943298, -0.16669800877571106, -0.3058528006076813, -0.4359988570213318, 0.5479963421821594, -0.17046374082565308, -0.255664199590683, 0.2537083923816681, 0.5106953978538513, 0.5959609746932983, -0.0342525951564312, 0.3270612061023712, 0.44173958897590637, 0.18562524020671844, 0.04784586653113365, 0.4570547938346863, -0.43809935450553894, 0.2916117310523987, -0.6281865239143372, -0.028995880857110023, -0.5740996599197388, -0.3116261959075928, -0.27761000394821167, 0.09440228343009949, -0.0025283358991146088, -0.9261763095855713, 0.0320887416601181, -0.30059942603111267, 0.011491984128952026, 0.10278117656707764, -0.2830929756164551, -0.3972940444946289, 0.3813233971595764, 0.23345857858657837, -0.5059336423873901, 0.11524523794651031, -0.185674250125885, -0.04667012393474579, -0.19806209206581116, -1.0427511930465698, 0.03248880058526993, 0.2785922884941101, -0.7174896001815796, 0.654216468334198, -0.41073182225227356, -0.6033300161361694, 0.21138611435890198, -0.7727997303009033, 0.22668103873729706, 0.9175397753715515, 0.6360582709312439, 0.1733676940202713, 0.09027408063411713, -0.3124241828918457, 0.3942979574203491, 0.5564291477203369, -0.14162495732307434, 0.6066649556159973, 1.0435086488723755, -0.10866902768611908, -1.0668243169784546, -0.3369048237800598, -0.0076075829565525055, -0.3238024115562439, 0.30940675735473633, -0.4437888264656067, -0.17224693298339844, 0.976441502571106, 0.1361335963010788, -0.5316841006278992, 0.09187886118888855, -0.5360389351844788, 0.7558504343032837, 0.5863571166992188, 0.21720664203166962, -0.3112919330596924, -0.08524750173091888, 0.5212619304656982, 0.32852333784103394, -0.4647563695907593, -0.06942272186279297, -0.8029506206512451, -0.3078831136226654, 0.15605033934116364, -0.46962058544158936, 0.5632956027984619, -0.7837184071540833, -0.14699779450893402, -0.7516410946846008, -0.12543582916259766, -1.3474907875061035, 0.8268484473228455, -0.42042407393455505, 0.7789855599403381, 0.3206159472465515, 0.14291846752166748, -0.8829846978187561, 0.11485592275857925, -0.15569935739040375, -0.6395000219345093, -0.7563928365707397, -0.140071839094162, 0.6056751608848572, 0.2809056043624878, -0.24478662014007568, 0.0682874396443367, 0.021173780784010887, 0.2669753432273865, -0.4586006700992584, 0.45752522349357605, 0.46422144770622253, 0.5764248371124268, -1.0596977472305298, -0.29120638966560364, 0.5497267842292786, 0.09579293429851532, -0.3347860276699066, 0.24167422950267792, -0.34228819608688354, 0.2279307097196579, -0.5476659536361694, -0.8244292140007019, -0.2505229115486145, -0.1525060534477234, -0.07239606976509094, 0.7113796472549438, -0.2938428223133087, -0.3747391402721405, -0.03314792364835739, -0.21986064314842224, -0.6922726631164551, 0.06669004261493683, -0.22076348960399628, 0.2218201607465744, -0.24249324202537537, -0.03342204913496971, 0.06746873259544373, 0.2254125326871872, -0.11362946033477783, 0.03245493397116661, -0.16911403834819794, -0.30464479327201843, -0.16107149422168732, -0.5277032852172852, 0.3635665774345398, -0.3059881329536438, -0.09504319727420807, 0.26927778124809265, 0.014523202553391457, 0.09811128675937653, 0.11730921268463135, 0.021787390112876892, 0.10563663393259048, -0.24218986928462982, -0.4074323773384094, -0.669375479221344, -0.04436112940311432, 0.02842221036553383, -1.166905403137207, -0.3280099630355835, 0.5390718579292297, -0.9953498840332031, 0.01933206059038639, -0.476377934217453, 0.3640170693397522, -0.1177491620182991, -0.020892301574349403, -0.4226142168045044, -0.23347750306129456, -0.28483307361602783, 0.22030097246170044, 0.3722095489501953, -0.5766949653625488, 0.1496526598930359, 0.6007220149040222, -0.930453896522522, 0.0081008430570364, 0.289975106716156, -0.33985331654548645, 0.17316414415836334, -0.13028469681739807, -0.8393482565879822, -0.17800772190093994, -0.4146674871444702, -0.25635677576065063, -0.8517197966575623, -0.25943130254745483, -0.49763137102127075, 0.4966796636581421, -0.6567263007164001, -0.05373726412653923, 0.10802126675844193, -0.6275954842567444, 0.13967962563037872, -0.5451675057411194, 0.07076438516378403, 0.40680116415023804, -0.10167185962200165, -0.5032976865768433, 0.5153709650039673, -0.013189436867833138, -0.1731434017419815, -0.04992056265473366, 0.1392611563205719, -0.2448924034833908, 0.11122235655784607, 0.5778167843818665, 0.960648238658905, -0.569469153881073, -0.4331369996070862, -0.07738300412893295, -0.12381561845541, 0.24445736408233643, 0.30232733488082886, -0.08863390982151031, 0.07335564494132996, -0.42377379536628723, 0.5532171726226807, 0.7121695280075073, 0.7779272794723511, -0.3005582392215729, 0.18006519973278046, -0.2213151752948761, -0.3678132891654968, -0.0968625396490097, -0.033376384526491165, -0.15753903985023499, -0.051092326641082764, -0.3561515510082245, 0.30882880091667175, -0.35817769169807434, 1.0139434337615967, -0.06903750449419022, -0.029396578669548035, 0.2987140119075775, 0.28413593769073486, 0.09099243581295013, -0.7563184499740601, 0.24309441447257996, 0.5373661518096924, 0.02729465439915657, -0.4777972102165222, -0.6100865006446838, -0.3520562946796417, -0.005494670942425728, -0.23413744568824768, 0.8392400741577148, 0.030709508806467056, -0.10740286111831665, -0.024635188281536102, -0.17153191566467285, 0.05560019612312317, -0.13033661246299744, -0.542241632938385, -0.4734562039375305, -0.49353551864624023, -0.5935662388801575, -0.7589391469955444, -0.1313403695821762, 0.5720183253288269, -0.40219274163246155, 0.06959269940853119, 0.37152326107025146, -0.37562257051467896, -0.02954629808664322, 0.930964469909668, 0.42209678888320923, 0.12162351608276367, 0.41103601455688477, 0.10832429677248001, -0.5678329467773438, 0.19651685655117035, -0.4050275385379791, -0.7527039647102356, 0.09162810444831848, 0.19681930541992188, -0.3514464199542999, 0.020000498741865158, -0.6124599575996399, 0.040947698056697845, 0.6245061755180359, 0.1486847847700119, -0.2900201976299286, 0.4328189194202423, 0.5073332190513611], "sim": 0.46454939246177673, "freq": 35.0, "x_tsne": 9.196413040161133, "y_tsne": -23.22793197631836}, {"lex": "pcv13", "type": "nb", "subreddit": "conspiracy", "vec": [0.05013042315840721, 0.16771207749843597, 0.38468462228775024, -0.08505186438560486, 0.20950870215892792, -0.05941876024007797, 0.03794408589601517, 0.0644737109541893, 0.3994550406932831, 0.24164950847625732, -0.1166001483798027, -0.012946833856403828, 0.13967463374137878, -0.03570638224482536, 0.13870202004909515, 0.17488060891628265, -0.07610397040843964, -0.17562492191791534, 0.07332037389278412, 0.029688561335206032, 0.09095645695924759, 0.1447562724351883, 0.1504865139722824, 0.033958423882722855, 0.18279549479484558, -0.11464172601699829, 0.16758282482624054, 0.16764281690120697, -0.05905107036232948, -0.13819058239459991, -0.15192784368991852, -0.2685166597366333, 0.045319296419620514, -0.2618463337421417, -0.07971752434968948, 0.20772993564605713, 0.05914684385061264, 0.008458476513624191, -0.08086688816547394, 0.21257899701595306, -0.07454773038625717, 0.011671718209981918, 0.04363156855106354, 0.04545256122946739, 0.0007992498576641083, 0.009808670729398727, 0.30955952405929565, 0.0004980992525815964, 0.18522575497627258, 0.09068812429904938, 0.09384006261825562, -0.3805617094039917, -0.07181751728057861, 0.11160953342914581, -0.10871905833482742, -0.006128937471657991, 0.053711358457803726, 0.18921908736228943, 0.3173506557941437, -0.14863398671150208, -0.021392345428466797, -0.3149312138557434, -0.035606831312179565, 0.16589972376823425, 0.18591362237930298, 0.2640262246131897, 0.019329536706209183, -0.09937345236539841, 0.03207189589738846, 0.24330444633960724, 0.029461221769452095, -0.14011414349079132, 0.0011537326499819756, -0.12237454205751419, -0.04395260661840439, 0.001155453035607934, -0.013399835675954819, -0.09507925063371658, -0.16344192624092102, -0.21097594499588013, 0.06582542508840561, -0.21576663851737976, -0.10113078355789185, 0.002919702557846904, -0.25500860810279846, -0.08220293372869492, -0.08799256384372711, -0.1255771368741989, 0.10318922251462936, -0.005067013204097748, -0.13679516315460205, 0.06265716254711151, 0.26485714316368103, -0.08961290121078491, 0.03365479037165642, -0.01988142356276512, -0.029445499181747437, -0.14832012355327606, 0.07979343086481094, 0.07604685425758362, 0.13770648837089539, 0.10275012254714966, 0.19339394569396973, 0.04812714830040932, -0.15600843727588654, 0.02231597527861595, -0.07651002705097198, -0.0044835396111011505, 0.10324002057313919, 0.2847556173801422, -0.23575231432914734, -0.3164496421813965, 0.02898040972650051, 0.02275196835398674, -0.27415746450424194, -0.0013634662609547377, -0.09223700314760208, -0.051265817135572433, 0.16323444247245789, 0.003739866428077221, -0.017862923443317413, 0.060385413467884064, 0.007512602023780346, -0.03593180328607559, -0.13225626945495605, 0.29850372672080994, -0.03577494993805885, -0.0015739132650196552, -0.010478478856384754, 0.26682859659194946, 0.16114766895771027, 0.0778854638338089, -0.21526993811130524, -0.2402561902999878, 0.07230773568153381, -0.005508230999112129, -0.07509099692106247, -0.0911409854888916, -0.25911062955856323, -0.026605164632201195, -0.05310467630624771, -0.020173443481326103, -0.030522800981998444, 0.10890400409698486, 0.1558409333229065, -0.23297567665576935, -0.10864300280809402, 0.10620088875293732, 0.008068845607340336, -0.11119935661554337, 0.09624451398849487, -0.12731435894966125, -0.29425567388534546, -0.03962364420294762, -0.0702647790312767, -0.3174245059490204, -0.050471626222133636, 0.03504094108939171, -0.1735258549451828, 0.1719774305820465, 0.15091551840305328, 0.2442779541015625, -0.2155664712190628, 0.3229301869869232, 0.023921877145767212, -0.3052513897418976, -0.33232712745666504, 0.20913167297840118, 0.06285594403743744, 0.3753969967365265, -0.04684560000896454, -0.051266662776470184, -0.006154644303023815, -0.14081719517707825, 0.0421237051486969, -0.07805675268173218, -0.01042239647358656, -0.0491332933306694, -0.05576213821768761, -0.07695387303829193, -0.24703525006771088, 0.15655355155467987, 0.08790247142314911, -0.09841370582580566, -0.024434641003608704, -0.42434313893318176, 0.20534813404083252, -0.12101546674966812, -0.09782398492097855, -0.027406340464949608, -0.18829187750816345, -0.03707239776849747, -0.10007975995540619, -0.10948352515697479, 0.12757417559623718, -0.14174415171146393, -0.16745585203170776, -0.10904664546251297, 0.02307816594839096, 0.12527860701084137, -0.08939886093139648, 0.33492276072502136, -0.03389750048518181, 0.058600977063179016, -0.01441924273967743, -0.03306039795279503, -0.07625173777341843, -0.023136086761951447, 0.11818130314350128, -0.0022120079956948757, 0.20357036590576172, -0.19472268223762512, -0.12884436547756195, 0.1001695767045021, 0.14468808472156525, 0.14369265735149384, -0.15649297833442688, -0.27392521500587463, -0.09745077788829803, -0.2907635271549225, 0.2077866643667221, 0.08837205171585083, 0.059022244065999985, -0.059211309999227524, -0.18771447241306305, -0.07312654703855515, -0.0565466433763504, 0.1817198395729065, 0.06606288254261017, 0.11890159547328949, 0.22740280628204346, 0.022397082298994064, -0.0035182470455765724, 0.3231598734855652, 0.30125510692596436, -0.11255737394094467, -0.15900678932666779, 0.44843021035194397, -0.021694593131542206, -0.018243225291371346, -0.0256359800696373, 0.005963749717921019, 0.10589883476495743, -0.18862368166446686, 0.10402045398950577, 0.15356752276420593, 0.02746085450053215, 0.07486378401517868, -0.03848343342542648, 0.21033455431461334, -0.1584041267633438, 0.07251901179552078, 0.15960930287837982, 0.40029165148735046, 0.11966641992330551, 0.019886137917637825, 0.03135310858488083, -0.14919224381446838, -0.0589166097342968, 0.02535092458128929, 0.11242683976888657, 0.02111181803047657, -0.12064647674560547, -0.3403095006942749, -0.020957939326763153, 0.0830809697508812, -0.0070098284631967545, -0.11754666268825531, 0.15617774426937103, 0.10698220133781433, -0.21740654110908508, -0.20893646776676178, -0.2502969801425934, 0.025071000680327415, -0.11652181297540665, 0.11144028604030609, -0.23528625071048737, 0.1313886195421219, -0.40555262565612793, -0.08426547050476074, -0.030793659389019012, -0.0458231158554554, 0.13731323182582855, -0.10625885426998138, 0.17967157065868378, -0.029427897185087204, 0.15223625302314758, -0.25165197253227234, -0.16310881078243256, 0.0727243646979332, 0.1807025820016861, 0.21227490901947021, 0.06585197150707245, 0.1410021185874939, 0.15093471109867096, 0.2454506903886795, -0.13467015326023102, 0.05985681712627411, -0.0830916240811348, -0.20031948387622833], "sim": 0.4522269070148468, "freq": 16.0, "x_tsne": 1.311928629875183, "y_tsne": -7.965399265289307}, {"lex": "hpv", "type": "nb", "subreddit": "conspiracy", "vec": [-0.056178364902734756, -0.005944310687482357, 2.044489860534668, 0.25247660279273987, 1.5062042474746704, 0.79625403881073, -0.7198969125747681, -0.6041814684867859, -0.021492287516593933, 2.3732948303222656, -0.8863263726234436, -1.1656734943389893, 1.2574807405471802, -1.1260480880737305, 1.3195828199386597, 0.6390529274940491, -0.7828302383422852, 0.19736118614673615, 0.25562402606010437, 4.083868503570557, -0.8327741622924805, 1.0454398393630981, -2.4441277980804443, 0.05627097189426422, 1.5563875436782837, -0.20069384574890137, -0.22252856194972992, 0.285540908575058, -0.5330020189285278, -0.7524176836013794, -0.4681636393070221, -3.373490810394287, 0.7182725667953491, -1.3984776735305786, 0.06846024096012115, 0.21815995872020721, -1.222393274307251, 0.5880913138389587, -0.7863155603408813, 0.6838891506195068, -2.8224239349365234, 1.2000762224197388, -0.9058568477630615, 0.7818129658699036, -0.17107565701007843, -0.4679633378982544, 1.8364259004592896, 0.2867567539215088, 1.8384569883346558, -0.51236891746521, 1.6778271198272705, -2.3159141540527344, 1.1847835779190063, 1.3403798341751099, -0.7658624649047852, -0.8586767911911011, -0.6953971982002258, 0.8665379285812378, 1.0062849521636963, -1.9704233407974243, 0.5043432712554932, 1.511482834815979, 0.4042665362358093, 1.2457267045974731, -0.8128075003623962, 0.6222389340400696, 0.3296387791633606, -0.8264749050140381, -1.1231917142868042, 2.8434479236602783, 0.8503850102424622, 0.384300172328949, -1.8319143056869507, -0.6596351265907288, -1.3252736330032349, 0.5730479955673218, 0.7619293928146362, -1.0624977350234985, -0.947300910949707, -0.34437429904937744, 0.7724940776824951, 0.6917891502380371, -2.5185558795928955, -0.5434858202934265, -0.9733561277389526, 0.12344755977392197, -1.6005887985229492, -1.2676926851272583, 0.6441188454627991, -0.9824429154396057, -1.1501328945159912, 0.17741069197654724, 0.8367470502853394, 0.046983759850263596, -0.10962957888841629, -1.2629002332687378, -0.8257549405097961, 2.312113046646118, 0.7276535630226135, 2.5755016803741455, 2.2805163860321045, -0.3850656747817993, 2.5778470039367676, -0.04965488612651825, -2.7715933322906494, 1.3986902236938477, -1.0983119010925293, -1.8531285524368286, 0.5449483394622803, 1.279638409614563, -1.5392556190490723, -0.2493959665298462, -0.35906705260276794, -1.0577545166015625, -1.4466650485992432, 0.398121178150177, -0.7120090126991272, 0.18599456548690796, -0.18234720826148987, -0.20130546391010284, -2.3710522651672363, -0.8085026741027832, -0.032060861587524414, 1.7731126546859741, -0.42877525091171265, 3.340090274810791, 0.39261215925216675, -0.6337852478027344, 0.3088209629058838, 2.1843740940093994, -0.657096803188324, 1.1578257083892822, 1.426621913909912, -1.0129756927490234, 0.7382269501686096, -0.4453866183757782, -1.035294771194458, -2.2227065563201904, 0.9026004076004028, -0.19804473221302032, -0.38230547308921814, -1.265616536140442, 0.023420361801981926, -0.34775927662849426, 0.9806646704673767, -1.654036283493042, -0.3918638229370117, -1.7440109252929688, 0.673940122127533, -0.7978177070617676, 0.652288556098938, -0.2965553104877472, -1.3998734951019287, -0.7200436592102051, -1.4438031911849976, -1.1031357049942017, -1.0420012474060059, -0.318943053483963, -2.5195095539093018, 1.9658710956573486, -0.5331505537033081, 1.4188902378082275, -1.5209004878997803, -1.1839327812194824, -0.8334569931030273, -1.2466062307357788, 1.608743667602539, 0.6734972596168518, 1.4231903553009033, 2.7647571563720703, -1.8763647079467773, 0.15623308718204498, 1.2939226627349854, -0.14170722663402557, -0.38794562220573425, -1.4917352199554443, 0.06933135539293289, -0.5426013469696045, -1.8230434656143188, -0.34658196568489075, -0.3572237193584442, 0.6872398257255554, -0.13288037478923798, -0.4523267447948456, -1.3323895931243896, 0.502050518989563, 0.5924956798553467, -0.8442529439926147, 1.0145467519760132, 0.34309008717536926, -0.8037158846855164, -2.1371989250183105, -1.5508692264556885, -1.1783850193023682, 2.397439956665039, 0.04037502035498619, -0.10729789733886719, -1.3518242835998535, 1.3535083532333374, 0.4308129847049713, -0.9819257259368896, -0.37983718514442444, -1.0708752870559692, -0.5218669176101685, -1.208938479423523, -0.7206498980522156, -0.6192693710327148, 0.10689746588468552, -0.04808136075735092, 1.9343311786651611, -0.5644006729125977, -2.21685791015625, -0.9692208170890808, 1.8409112691879272, 0.39676687121391296, -0.13872984051704407, -0.43415066599845886, -1.3196486234664917, 0.39281660318374634, 0.28877416253089905, 2.276036024093628, 0.8942806124687195, 0.11839942634105682, -0.7044020891189575, -1.525067925453186, -0.5162959098815918, 0.34151920676231384, 0.7101386785507202, -0.47095897793769836, -0.39421114325523376, 1.6718097925186157, 0.18873655796051025, 0.12752847373485565, 0.8953336477279663, 1.9138453006744385, 0.38153547048568726, 1.016914963722229, 1.1299182176589966, -1.1661326885223389, -0.8569169044494629, -0.9173556566238403, -1.776436686515808, 0.5024473667144775, -0.575664222240448, -0.9999650716781616, -0.7796741724014282, -0.38560596108436584, -0.47058776021003723, 1.6290229558944702, -1.1939194202423096, 0.6977700591087341, -1.6124777793884277, 2.3859786987304688, 2.626924753189087, 0.8210054039955139, 1.571230173110962, -0.11905281245708466, -0.20239369571208954, 1.1498461961746216, 2.0629451274871826, 1.1306761503219604, 1.242517113685608, -2.5995965003967285, -2.2846477031707764, -0.5928850173950195, -2.138056993484497, -0.5219843983650208, 1.5355571508407593, 0.7563669681549072, 0.7579042911529541, 0.4712793231010437, -4.263016700744629, -1.3524904251098633, 0.7749417424201965, 1.2413712739944458, 2.4326400756835938, -0.8998715877532959, 0.37929099798202515, -0.04607859626412392, -0.24173128604888916, -1.1585032939910889, 1.0926733016967773, -1.0871378183364868, -1.480189323425293, -0.8446751236915588, -0.2633272111415863, 1.9254511594772339, 0.1538384109735489, 0.5783734321594238, -0.17122894525527954, 2.2643537521362305, 0.020756784826517105, -0.3253447711467743, 0.7269539833068848, -1.9344069957733154, -1.3308510780334473, -1.8814157247543335, 2.258451461791992, -2.0094356536865234, 0.6102017164230347], "sim": 0.451971173286438, "freq": 699.0, "x_tsne": -1.3356389999389648, "y_tsne": -24.280988693237305}, {"lex": "chickenpox", "type": "nb", "subreddit": "conspiracy", "vec": [0.49554553627967834, 0.09887804836034775, 1.1258409023284912, -0.3992302715778351, -0.13423816859722137, -0.3597436547279358, 0.11169048398733139, -0.6325822472572327, 0.8250241875648499, 1.372212290763855, -0.35607680678367615, -0.7767440676689148, 0.9307786822319031, -0.7835601568222046, -0.42903876304626465, 0.7716455459594727, 0.4074617326259613, -1.2663190364837646, -0.00015419721603393555, 0.7542645931243896, -0.4797120690345764, 0.5618487000465393, 0.6177018880844116, 0.8655860424041748, 0.9841479063034058, -0.031465742737054825, 0.7369718551635742, 0.459404319524765, 0.6976931691169739, 0.2526293396949768, -0.7125247120857239, -1.4886398315429688, 0.12507209181785583, -1.1752970218658447, -0.6551700830459595, 1.5451908111572266, -0.1959337592124939, -0.4290046989917755, 0.45356476306915283, 2.1828489303588867, -0.4837152063846588, 1.7120288610458374, -0.49235567450523376, 1.6037402153015137, -0.3760843873023987, 0.6872843503952026, 0.37717515230178833, -0.961768627166748, 0.9029296636581421, 0.5403991937637329, -0.05469382181763649, -1.133622407913208, 0.2611362040042877, 0.7519472241401672, -0.5216686725616455, -0.49890491366386414, 0.44420725107192993, 0.37225812673568726, 1.1349941492080688, -0.08385148644447327, 0.8365889191627502, 0.8734702467918396, -1.007946491241455, -0.6339019536972046, 1.1007144451141357, 0.21830782294273376, 0.7294014692306519, 1.0859349966049194, -0.4306812882423401, 1.397996425628662, 1.5750905275344849, -0.9503263235092163, -0.8999453783035278, -0.27916866540908813, -1.6308726072311401, -0.08770564198493958, 0.6824543476104736, -0.4756419360637665, -1.3516710996627808, -0.009849844500422478, -0.5214913487434387, -0.43231910467147827, -0.566357433795929, -0.6878114342689514, -0.3492361009120941, 0.42994600534439087, -1.7419999837875366, -0.7376194000244141, -0.6877956986427307, 0.8127350807189941, -0.1575930416584015, 0.24358513951301575, 0.05066747963428497, 0.8120930194854736, -0.5035592317581177, -1.105173110961914, 0.2973547875881195, -0.2797665297985077, 0.3105345666408539, -0.3777868449687958, 0.24213960766792297, -0.4779680669307709, 0.18409255146980286, -0.19891779124736786, -1.233600378036499, -0.23758645355701447, -0.14574408531188965, 0.15364421904087067, 0.9340770840644836, -0.3113815188407898, -0.3426979184150696, -0.8278653025627136, 0.5851497054100037, 0.12920048832893372, -0.49957066774368286, -0.2745307981967926, -0.08605632185935974, 0.9367332458496094, 0.23920029401779175, 0.08664722740650177, -0.5809868574142456, -0.5832481384277344, 1.1724522113800049, 0.016496770083904266, -0.013509206473827362, 1.2944046258926392, -0.06922595947980881, -0.06910532712936401, 0.5183839797973633, 1.1130791902542114, -0.5638543963432312, -1.290613055229187, 0.49745631217956543, -0.2053651660680771, 0.6042003035545349, 0.46949058771133423, -1.6828651428222656, -1.0470179319381714, 0.5937336683273315, -0.15597029030323029, -0.9211435914039612, -0.7001113891601562, -1.3764020204544067, 0.47325342893600464, 0.3555910289287567, -1.2458802461624146, -0.16865494847297668, -1.2758651971817017, 0.8026508688926697, -0.6590031981468201, -0.026882776990532875, 0.7982209920883179, -1.7742499113082886, 0.9226354360580444, -0.4622846245765686, -0.06435789912939072, -0.44120272994041443, 0.863600492477417, -1.0760356187820435, 1.2156835794448853, 0.6733178496360779, 1.2658337354660034, -0.4443308711051941, 0.5422454476356506, 0.2909732758998871, -0.8273410201072693, -1.5435270071029663, 0.443082720041275, 1.6433172225952148, 1.598970890045166, -0.8893766403198242, 0.3199731111526489, 0.7722694873809814, -0.12309330701828003, 1.0135763883590698, -0.2830093502998352, -0.7062519192695618, -0.776536762714386, -1.4382976293563843, 1.0211137533187866, -1.4869842529296875, 0.7132508754730225, 0.9607331156730652, -1.0286965370178223, -0.24471373856067657, -0.8377988338470459, 0.9884678721427917, -0.363756388425827, 0.4390427768230438, 0.4666239321231842, -0.21455895900726318, -0.5010616779327393, 0.31328892707824707, 0.38919389247894287, 0.8270102739334106, 0.16932617127895355, -0.24632716178894043, 0.4272446036338806, 0.026502177119255066, 0.22215278446674347, -0.2918301224708557, 1.2930963039398193, -0.2858145833015442, 0.2099263072013855, -0.9364526271820068, -0.45563390851020813, -0.29479485750198364, -0.4361516833305359, 0.9075093269348145, 2.0167503356933594, 0.027163727208971977, -1.5616953372955322, 1.0385856628417969, 0.7118415236473083, -0.7643766403198242, 0.10723787546157837, -0.546079158782959, -0.8096508979797363, -2.2829020023345947, 0.418967068195343, 1.5872302055358887, -1.030293583869934, 0.5089215636253357, -1.5191551446914673, -0.17591606080532074, 0.3791715204715729, 0.39207497239112854, 0.6391263604164124, -0.19839036464691162, -0.28847426176071167, 2.595385789871216, 0.28691354393959045, 0.9159888029098511, 1.589674949645996, 2.0652050971984863, -1.0506948232650757, 0.5907155871391296, -0.8890738487243652, 0.5395732522010803, -0.42317113280296326, -0.08010829985141754, -1.0439468622207642, -0.019257668405771255, -0.060645002871751785, -0.3028881847858429, -0.8706547021865845, 1.9261373281478882, -0.5771217942237854, 1.2411562204360962, 0.2618807256221771, -1.5692600011825562, -0.6555920839309692, 0.9143146872520447, 1.202486276626587, 0.5278298854827881, 1.4614912271499634, -2.0795505046844482, -0.3670235276222229, -1.3870048522949219, 1.9234647750854492, 1.3792725801467896, -0.5936992764472961, -0.36461082100868225, -1.7429043054580688, -0.6529768705368042, -0.9276187419891357, -0.03978479653596878, 0.2085828185081482, 0.9993051886558533, 0.426521897315979, 0.7548964619636536, -2.614424228668213, -1.2918531894683838, -0.23205839097499847, 0.4041016399860382, 0.6872695684432983, -0.45265084505081177, 3.3394649028778076, -0.8621115684509277, -0.3052871823310852, -0.6591355204582214, -0.08720564842224121, -0.217227503657341, 0.32644975185394287, 1.8136529922485352, 0.49754583835601807, 0.9383816123008728, -0.8132916688919067, 0.685173511505127, 0.7821091413497925, 1.8796957731246948, 0.4048258364200592, -0.02907603606581688, -0.009775991551578045, 0.2595440447330475, 0.9574202299118042, -1.0636314153671265, 0.5217031240463257, -1.3618303537368774, -0.979098379611969], "sim": 0.4515800476074219, "freq": 437.0, "x_tsne": -8.992395401000977, "y_tsne": -26.565004348754883}, {"lex": "immunisation", "type": "nb", "subreddit": "conspiracy", "vec": [0.008262119255959988, 0.6681673526763916, 1.3155597448349, -0.7006620764732361, 1.195757269859314, -0.8081247210502625, 0.08044906705617905, -0.3014046251773834, 0.8296099305152893, 0.2759036719799042, 0.7896298170089722, -0.536963164806366, -0.17619280517101288, -1.1152770519256592, 1.3120397329330444, 0.8074665665626526, -0.34978970885276794, -0.1656821370124817, -0.0909646600484848, 0.8379002809524536, 0.5783258676528931, -0.30591389536857605, -0.2945813238620758, 0.15694057941436768, 0.5524036884307861, -0.22881241142749786, -0.21486468613147736, -0.19051799178123474, -0.906767725944519, -0.28680020570755005, 0.6772019863128662, -0.3503558039665222, 0.8921652436256409, -0.8791507482528687, 0.44033998250961304, -0.4159433841705322, -0.6579806208610535, 0.4709785282611847, -0.4177291691303253, 0.7843701839447021, -0.9947910904884338, 0.9894412755966187, -0.8797249794006348, 0.22122572362422943, 1.4375805854797363, -0.6499701142311096, 0.956500768661499, 1.1562530994415283, 0.8010370135307312, 0.2741490304470062, 0.8230751752853394, -0.8144416809082031, -1.1688265800476074, 0.9801463484764099, -0.21846316754817963, 0.3179023861885071, 0.2379627227783203, 0.5997183918952942, 0.375521183013916, -0.24961332976818085, 0.18597161769866943, 0.672552227973938, -0.4271785020828247, 0.8110727667808533, -0.6260712742805481, 0.2209172546863556, -1.5263514518737793, 1.065304160118103, -0.09971063584089279, 0.607492208480835, -0.34268718957901, -1.1654130220413208, -0.5847219824790955, -0.07475605607032776, -0.41264966130256653, -0.45065292716026306, -0.1956302523612976, 0.33123743534088135, -1.5821564197540283, -0.24685965478420258, 0.7373261451721191, 0.21381966769695282, -0.6760786175727844, -1.117722511291504, -0.6939583420753479, 0.6116251349449158, -0.5198299288749695, -0.903863251209259, 0.49151232838630676, 0.2904408872127533, -0.2020227313041687, 0.04034065827727318, 1.58415687084198, 0.8584713339805603, 0.29882878065109253, -0.7638735175132751, 0.5015133619308472, -0.41880372166633606, -0.3201991319656372, -0.5141393542289734, 0.4535444378852844, -0.8362976908683777, -0.43892285227775574, -0.1461561918258667, -0.5638964176177979, -0.35202887654304504, 0.6487732529640198, -0.09937000274658203, 0.5787443518638611, 0.5161684155464172, -0.743777334690094, -0.9613745808601379, -0.4857899248600006, 0.6425116658210754, -0.9985078573226929, -0.22260555624961853, 0.21909524500370026, 0.34986430406570435, 0.24077782034873962, -0.1725654900074005, -1.0236505270004272, -0.2719661593437195, -0.4699615240097046, 0.6842065453529358, -0.2511261999607086, 0.5714507102966309, 0.46033546328544617, -0.3827624022960663, -0.21945035457611084, 1.3955482244491577, 0.00582633912563324, -0.16943126916885376, 0.2766508460044861, 0.016324495896697044, -0.13409541547298431, -0.6723538041114807, -0.8778077363967896, -0.96785569190979, -0.8527709245681763, -0.9966971278190613, -0.009263616986572742, -0.5870348811149597, -0.2964211702346802, 0.4562605917453766, 1.8613166809082031, -1.8342045545578003, 0.2840612828731537, -0.07980132102966309, 0.3717668652534485, 0.47049760818481445, 0.015158940106630325, 0.4848136305809021, -1.804618239402771, -1.0611413717269897, -0.7467725276947021, -0.07403580844402313, -0.031528789550065994, -0.17044973373413086, -1.5995023250579834, -0.4364803731441498, -0.33433711528778076, -0.09865452349185944, -0.4926055073738098, -0.43424805998802185, 1.249922513961792, -0.9197475910186768, -0.4761068522930145, 1.717253565788269, -0.015162684954702854, 0.6424786448478699, -0.3544359803199768, 0.04424867406487465, 0.023694870993494987, -0.11936128884553909, -0.6389233469963074, -0.06459803879261017, 0.9304769039154053, -0.21279485523700714, -0.5978785157203674, 0.49897146224975586, 0.2706255614757538, 1.378969669342041, 0.019707966595888138, -0.19057470560073853, -0.1522049754858017, -0.7222597002983093, 0.07854931056499481, -0.23315562307834625, 0.026690149679780006, 0.10397759824991226, -1.2424551248550415, 0.32046008110046387, -0.10004167258739471, -0.8823996782302856, 0.5256575345993042, 0.14891216158866882, 0.734843909740448, -0.9740166664123535, 0.465643048286438, 0.27628663182258606, -0.5649912357330322, -0.3121107518672943, -0.8112785220146179, 0.8892973065376282, -1.3617080450057983, 0.13432295620441437, -0.16535797715187073, -0.3340204358100891, -0.021186787635087967, 0.4174324572086334, -0.012638537213206291, 0.3773650527000427, 0.3221696615219116, 1.0768189430236816, 0.2454194277524948, 0.20919036865234375, -0.2495027333498001, -0.47062966227531433, 0.2857658863067627, -1.118675708770752, 0.1683925986289978, -0.25901612639427185, -0.04793323576450348, -0.8284712433815002, -1.2250629663467407, 0.04469548538327217, -0.9010030627250671, -0.5788366794586182, -0.11805891990661621, -0.09040813893079758, 1.0628509521484375, 0.008648957125842571, -0.2580341398715973, 0.5926162004470825, 0.6350249648094177, -0.17929422855377197, -0.8281122446060181, 1.205046534538269, -0.248173788189888, 0.1664135754108429, 0.09727749228477478, 0.061539169400930405, -0.06918758153915405, -0.35064005851745605, -0.4029196500778198, 0.9606220126152039, 0.6605125665664673, -1.4765387773513794, 1.3802424669265747, -0.9470112323760986, 0.29935067892074585, -0.48199233412742615, 1.1406623125076294, 1.6773662567138672, 0.7071841955184937, -0.3084459900856018, 0.07297084480524063, 0.16023939847946167, 0.678554356098175, -0.0673547238111496, 0.6796730756759644, 0.037588346749544144, -0.8042634129524231, -0.06558658182621002, -0.8401075005531311, -0.44959473609924316, -1.04691481590271, 0.17072956264019012, 0.17643098533153534, 0.5731553435325623, 0.0021926118060946465, -1.5844241380691528, -1.2910505533218384, 0.5811636447906494, -0.3464391827583313, 0.32423919439315796, 0.41347333788871765, 1.8516806364059448, -0.19789521396160126, 0.7078929543495178, -0.9793155789375305, 0.005028548650443554, -1.1969664096832275, 0.8994648456573486, 0.8012233376502991, -0.2548712491989136, -0.18787753582000732, -0.47583499550819397, -0.039862260222435, 0.037478651851415634, 0.8697924017906189, 1.13423490524292, 0.3232940137386322, 0.5899286270141602, 0.979148268699646, 0.08944116532802582, 0.2586541771888733, -0.1238851547241211, -0.043546196073293686, -0.602530837059021], "sim": 0.45117226243019104, "freq": 110.0, "x_tsne": 0.6186690926551819, "y_tsne": -21.356403350830078}, {"lex": "toxoid", "type": "nb", "subreddit": "conspiracy", "vec": [0.40653082728385925, 0.5471200942993164, 0.44141465425491333, -0.664251983165741, 0.045545127242803574, 0.3240819275379181, 0.04029162973165512, 0.4175313711166382, 0.33621522784233093, 0.2124682515859604, 0.16905289888381958, -0.3285103440284729, 0.6060735583305359, -0.42806801199913025, -0.06733206659555435, 0.3495272696018219, 0.0684981718659401, -0.4291217029094696, -0.09635207056999207, 0.6856933236122131, 0.2775636613368988, -0.2883985638618469, -0.40361487865448, -0.07022098451852798, 0.44232288002967834, -0.26014623045921326, -0.12775540351867676, 0.3906487822532654, -0.3235573470592499, 0.31872183084487915, -0.552361011505127, -0.47963035106658936, 0.01567496545612812, -0.5413251519203186, 0.23631304502487183, 0.6706733703613281, -0.40330496430397034, -0.13923251628875732, -0.7253583669662476, -0.10696151852607727, -0.1784423291683197, 0.008990665897727013, 0.3234034776687622, -0.37247687578201294, -0.19425474107265472, -0.14139901101589203, 0.9194090962409973, 0.2494390457868576, 0.45272907614707947, 0.10382901132106781, 0.16862641274929047, -0.47700396180152893, -0.3410225808620453, 0.26541030406951904, -0.1382916122674942, -0.30652034282684326, 0.0997965931892395, 0.026645643636584282, 0.22065412998199463, -0.39259567856788635, 0.4452936053276062, 0.1426243633031845, -0.16707846522331238, 0.3163096606731415, 0.07549145072698593, 0.28307050466537476, 0.14788435399532318, -0.1656774878501892, 0.14051498472690582, 0.056209124624729156, 0.5137520432472229, -0.13664378225803375, -0.43594688177108765, 0.08569328486919403, -0.2671680748462677, -0.2594127953052521, 0.034290578216314316, 0.049064960330724716, -0.09149929881095886, -0.10555668920278549, 0.21535015106201172, -0.32484865188598633, -0.6353645920753479, 0.039523229002952576, -0.3185299336910248, 0.15575416386127472, -0.2860978841781616, -0.1911257654428482, 0.6348795890808105, -0.08368620276451111, 0.4755721092224121, 0.46696966886520386, 0.49431049823760986, 0.017206892371177673, -0.1552300900220871, 0.11845976859331131, 0.1715724617242813, -0.05195727199316025, 0.15740707516670227, 0.2300414741039276, -0.13089196383953094, -0.37981075048446655, 0.15710124373435974, -0.03347526490688324, -0.8429493308067322, 0.19579723477363586, 0.04152567312121391, -0.18162986636161804, 0.3782899081707001, 0.09444171190261841, -0.5490374565124512, -0.6784955859184265, 0.24507391452789307, 0.03781997412443161, -0.42337366938591003, 0.08210981637239456, -0.05129775032401085, 0.05530548468232155, -0.21537859737873077, -0.09159855544567108, -0.08021116256713867, -0.10102605819702148, 0.008849920704960823, 0.40647995471954346, -0.27810198068618774, 0.8956781625747681, -0.7218781113624573, -0.014305204153060913, -0.10370335727930069, 0.5551363229751587, 0.27388647198677063, 0.18363520503044128, -0.5763019323348999, -0.14996547996997833, 0.3127368688583374, 0.1476803869009018, -0.5368857383728027, -0.3548213839530945, -0.3210068345069885, -0.1639060080051422, 0.07244393229484558, -0.2992916405200958, -0.11210888624191284, -0.17447715997695923, 0.08802115172147751, -0.030954066663980484, -0.2672644555568695, -0.2939276099205017, -0.2415584772825241, 0.3209318518638611, 0.12215811014175415, -0.128934845328331, -1.0895626544952393, 0.18442915380001068, -0.1746881902217865, -0.436000794172287, -0.09597576409578323, -0.30349525809288025, -0.6147716045379639, 0.440322607755661, 0.18209916353225708, -0.039469730108976364, -0.2999947667121887, 0.29065102338790894, -0.3687724471092224, -0.5117782950401306, -0.15082316100597382, 0.4443781077861786, 0.07237479090690613, 0.3925905227661133, -0.04087096452713013, 0.09618254750967026, -0.9467194676399231, -0.008344008587300777, -0.3577995002269745, -0.10780671238899231, -0.057334139943122864, -0.2849806845188141, -0.23888428509235382, -0.38179948925971985, 0.18622152507305145, 0.2394653856754303, -0.06689395755529404, 0.18143601715564728, -0.22168545424938202, -0.43358075618743896, 0.1971665471792221, -0.2787967324256897, 0.019967379048466682, 0.17193178832530975, -0.6705443263053894, 0.16475403308868408, -0.3368690609931946, 0.0564543753862381, 0.7200173735618591, -0.04835256189107895, 0.670021653175354, 0.36828741431236267, 0.2755277454853058, -0.1494138538837433, -0.661900520324707, 0.3207547664642334, -0.012435417622327805, 0.41344672441482544, -0.4149434566497803, 0.1696140468120575, -0.6120911240577698, 0.026356706395745277, -0.38563576340675354, 0.01855253055691719, -0.868174135684967, -0.358076810836792, 0.191819429397583, 0.46853405237197876, 0.26105672121047974, 0.059810616075992584, 0.5824048519134521, -0.3703073263168335, -0.39671725034713745, -0.646886944770813, 0.5262485146522522, 0.021227644756436348, 0.4820637106895447, 0.43064647912979126, -0.6394281983375549, -0.06169980764389038, 0.06651927530765533, 0.09513891488313675, -0.16106674075126648, -0.3448528051376343, 0.2999967932701111, -0.3327958583831787, -0.08219398558139801, 0.3348710834980011, 0.6657823324203491, -0.05197739228606224, -0.006588439457118511, 0.48199397325515747, -0.13263122737407684, -0.02054547518491745, -0.07519084215164185, -0.09132514894008636, 0.2383836805820465, -0.43545088171958923, -0.4031221866607666, 0.6440788507461548, 0.029299315065145493, -0.0032820142805576324, 0.17918819189071655, 0.14467310905456543, -0.3968701660633087, -0.33362287282943726, 0.529841959476471, 0.5830156207084656, 0.21581150591373444, 0.2105368822813034, -0.5694605708122253, 0.09060616046190262, 0.2507282495498657, 0.32162803411483765, 0.4053271412849426, 0.11143629252910614, -0.15455493330955505, -0.5538370609283447, -0.09896191209554672, -0.3938746452331543, -0.01921888254582882, -0.342120498418808, 0.31224361062049866, -0.30944082140922546, -0.3434808850288391, -0.7861555814743042, -0.15265825390815735, 0.28054672479629517, 0.4938899278640747, 0.3413337171077728, -0.1982770413160324, 1.4192107915878296, -0.09464184194803238, -0.13158929347991943, -0.14586541056632996, 0.2577304542064667, -0.04028461128473282, 0.09902747720479965, 0.2070372849702835, 0.2804190218448639, -0.017992112785577774, -0.2010669708251953, -0.04097788408398628, -0.3494553565979004, 0.7817850112915039, -0.04081740230321884, 0.22293050587177277, 0.627580463886261, -0.07271669805049896, 0.0769997090101242, -0.13738217949867249, 0.21486693620681763, -0.21878306567668915, -0.2167685627937317], "sim": 0.45112091302871704, "freq": 29.0, "x_tsne": -1.0400103330612183, "y_tsne": -18.517168045043945}, {"lex": "chemotherapy", "type": "nb", "subreddit": "conspiracy", "vec": [-0.41374310851097107, -2.2406060695648193, 1.146566390991211, -1.031740427017212, -0.11009535193443298, 0.5094835758209229, -0.5528992414474487, 1.2369968891143799, 0.8940085768699646, -0.2781544327735901, -0.253714919090271, -0.6821079254150391, 1.1742466688156128, -1.9767874479293823, 0.600894570350647, 0.010228779166936874, -0.006848864257335663, 0.5852817296981812, 1.2584160566329956, 1.1000165939331055, -1.2426282167434692, 2.300912618637085, 0.2403564155101776, -0.42105633020401, 1.8631826639175415, 0.5155115127563477, 1.468785047531128, 1.269213318824768, -0.2558403015136719, -0.07865297794342041, 0.16436553001403809, -0.5311266183853149, 1.4085246324539185, -0.747593879699707, 0.5603592395782471, 0.4843646287918091, -0.37564602494239807, -2.033726215362549, 0.2352062612771988, -0.2740305960178375, -0.6694343090057373, -1.1057778596878052, -0.6794416904449463, 1.140636920928955, 0.3631216585636139, -0.6511395573616028, 2.298103094100952, 1.0130558013916016, 0.3400869369506836, 0.5602703094482422, 0.6698299050331116, -0.25104355812072754, -0.33120742440223694, 0.6308569312095642, -1.9240062236785889, -0.4801609516143799, -0.9084029793739319, 1.0057600736618042, -0.1336180567741394, -0.6034610271453857, 0.5664705634117126, 0.39355337619781494, -0.9811794757843018, 1.124008297920227, 1.6653180122375488, 1.136731505393982, 0.7906015515327454, 0.025882460176944733, 0.2068081498146057, 2.3417844772338867, 0.8679922819137573, -1.1354113817214966, 1.0008701086044312, 0.37769144773483276, -1.46620774269104, -0.33666959404945374, -1.046297550201416, 0.4923139214515686, -1.1515451669692993, 0.3034968078136444, 0.18134401738643646, -0.19122463464736938, 0.4734618365764618, -0.487335741519928, -0.8734604716300964, 0.28598552942276, -0.04675379395484924, -2.315209150314331, 0.5655919313430786, 0.29350340366363525, -1.0464048385620117, 0.4662293493747711, 1.838139533996582, 0.06290338933467865, 0.6641762852668762, -0.8281858563423157, 1.065900206565857, -0.45773041248321533, -0.8806588053703308, 0.5919790267944336, 0.7354562878608704, 0.18982243537902832, -0.3206300735473633, -0.3281213045120239, -2.9088761806488037, -0.7279977202415466, 0.24128516018390656, 0.22864806652069092, -0.5641286969184875, 0.8194239735603333, 0.05085518956184387, -1.2687156200408936, -0.16632653772830963, 0.22273477911949158, -0.5012379884719849, 0.2656330168247223, -0.8908036351203918, 0.6018421053886414, 0.7155117392539978, -0.27783024311065674, 0.5505995154380798, -0.32489994168281555, 0.2992841899394989, 0.5891851782798767, -0.13348685204982758, -0.031513918191194534, 0.9003514647483826, -1.2980998754501343, 0.8923795223236084, 0.4881874620914459, -0.5306086540222168, -0.5739709734916687, 0.7650502920150757, 1.0092723369598389, 1.0526988506317139, -1.6129990816116333, -0.29986727237701416, -1.528831124305725, 1.0137068033218384, -0.015306845307350159, 0.76841139793396, -1.7190661430358887, -0.7347037196159363, 1.860643744468689, -1.0730986595153809, -2.276240825653076, 0.555889368057251, 0.16501528024673462, 0.9305741786956787, -1.5017776489257812, 0.08581969141960144, -1.438492774963379, 1.1961207389831543, -0.4049505293369293, -0.7675492763519287, -1.8877748250961304, -1.5452269315719604, 0.1235150620341301, -0.8144473433494568, 0.3624250888824463, -0.18315470218658447, 0.8955361247062683, -0.7570613622665405, 0.6114901900291443, -1.150589108467102, 0.11885937303304672, 0.17520835995674133, 0.6407940983772278, -0.0965869277715683, 1.9544529914855957, 0.4451074004173279, -0.6829826831817627, 1.5478414297103882, 0.7922312021255493, -0.3466499447822571, 0.2775926887989044, -1.0280169248580933, -0.7395769357681274, -1.4086220264434814, -0.24914924800395966, 0.048235177993774414, 1.184838056564331, -0.5330995321273804, 1.2140065431594849, -0.3688613474369049, 1.1692821979522705, 1.663633108139038, 0.8376717567443848, 2.3875951766967773, -0.5280167460441589, -0.45872414112091064, -1.4532349109649658, 0.36643078923225403, -0.04084128513932228, -0.16525159776210785, -0.5000864863395691, -1.0922945737838745, 0.8359248638153076, 1.9333478212356567, 0.6559401750564575, -3.0147509574890137, -0.5137182474136353, 0.30642473697662354, 3.4649734497070312, -0.7382140755653381, -0.6892111897468567, -0.10700367391109467, -0.7848380208015442, 1.3406375646591187, 0.46681052446365356, -0.7774965167045593, -1.4523155689239502, -0.5164026021957397, -0.01213034987449646, -0.8362927436828613, -1.3620343208312988, -1.0192652940750122, -0.536619246006012, 0.47343116998672485, -1.7368332147598267, -0.278685986995697, 1.2936984300613403, -0.04403781145811081, -0.058193858712911606, 0.7523596286773682, -0.2904690206050873, 0.4984319508075714, -0.9331164360046387, 0.46828824281692505, 1.9861503839492798, 1.562248945236206, 0.1788201630115509, -0.7419573068618774, 0.30480557680130005, 1.0666654109954834, 0.06386370956897736, 0.5300275087356567, 1.0215896368026733, 0.14432236552238464, -1.037250280380249, -0.1314694732427597, -0.12175354361534119, 0.7598816156387329, -0.30939334630966187, 0.7149645686149597, -0.7366422414779663, 0.23425154387950897, 1.1505728960037231, 0.9870882034301758, 0.9359018206596375, 0.6231906414031982, -1.4199870824813843, 0.1375722587108612, 1.681333303451538, 1.484940528869629, -0.4337422847747803, -1.7581026554107666, 1.3117265701293945, -0.37456199526786804, -0.5040108561515808, -0.13176947832107544, 0.20556709170341492, -0.2276873141527176, -1.7368074655532837, -1.3608648777008057, -2.2398433685302734, -0.4829588830471039, -0.21055260300636292, 1.2298719882965088, -0.17014583945274353, 1.2882946729660034, -1.7840272188186646, -0.47251376509666443, -0.4155442416667938, 0.13144272565841675, 0.7362273335456848, 0.2350340336561203, 1.8257864713668823, -1.0030287504196167, -0.1707824170589447, -1.7268799543380737, -2.1963350772857666, -0.8044615387916565, 0.04749376326799393, 0.60517817735672, -0.6830403208732605, 0.42939379811286926, -0.7043580412864685, 0.04555537551641464, 1.6292378902435303, 2.4521257877349854, 0.5783928036689758, 0.3973177671432495, 1.4968199729919434, 0.3021922707557678, -1.218440055847168, -0.6895259618759155, 1.0645633935928345, -0.20121541619300842, 0.4732137620449066], "sim": 0.44998782873153687, "freq": 331.0, "x_tsne": -35.40567398071289, "y_tsne": 14.416863441467285}, {"lex": "tdap", "type": "nb", "subreddit": "conspiracy", "vec": [-0.5987979769706726, 0.7151318192481995, 0.5236795544624329, -0.40691718459129333, 0.7893365621566772, -0.06228688359260559, -0.13739275932312012, -0.006494790315628052, 0.6093090176582336, 0.3502306044101715, -0.16484874486923218, 0.3136608898639679, 0.27400463819503784, -0.25798699259757996, 1.096001386642456, 0.4594491720199585, 0.0022284016013145447, -0.9976752996444702, 0.13244874775409698, 1.340624213218689, 0.1766778528690338, -0.15818578004837036, -0.03646164387464523, 0.6724562048912048, 0.2996029555797577, -0.2971624732017517, -0.7178768515586853, -0.01931982859969139, -0.4440564811229706, -0.15254616737365723, -0.5085760354995728, -0.48363107442855835, 0.18996326625347137, -0.5461864471435547, -0.08442021906375885, 0.4282727837562561, -1.0292762517929077, -0.30045709013938904, -0.372367262840271, 0.08892863243818283, -0.1863492876291275, 0.135423943400383, -0.39305588603019714, 1.0077053308486938, -0.07375601679086685, 0.18104849755764008, 0.10368964076042175, -0.26817017793655396, 0.9418266415596008, 0.04104974865913391, -0.5386065244674683, -0.6826009750366211, -0.2790624797344208, 0.2347874790430069, -0.7551816701889038, -0.000703694298863411, 0.5523374080657959, -0.02713112160563469, 0.4704782962799072, -1.3778748512268066, 0.4344097375869751, -0.5791805982589722, 0.016617856919765472, 0.8247332572937012, 0.4203631579875946, 0.47549349069595337, 0.11449803411960602, 0.17763707041740417, -0.44256800413131714, 1.378566026687622, 0.6876228451728821, -0.10008738189935684, -0.21712948381900787, -0.1465373933315277, -0.248010516166687, -0.030862931162118912, -0.6097399592399597, -0.5909239053726196, 0.5117471814155579, -0.3391115665435791, 0.7688266038894653, 0.0864119753241539, -0.5252432227134705, -0.401695191860199, -0.47560256719589233, 0.47448891401290894, -1.214935302734375, -0.03482113778591156, 0.12175004184246063, -0.30104953050613403, -0.2927916646003723, 0.47425898909568787, 0.5609880685806274, 0.2856944799423218, -0.25093188881874084, -0.3522895276546478, 0.7416448593139648, 0.31002458930015564, 0.6588585376739502, -0.09040001034736633, 0.21993008255958557, -0.05166495591402054, 0.1596793383359909, -0.5963876247406006, -0.8867054581642151, -0.6095278859138489, 0.7573951482772827, -0.2982199490070343, 0.8831611275672913, 0.12111727148294449, 0.03616887331008911, -0.8921240568161011, 0.541684627532959, 0.8462249040603638, -0.302916020154953, -0.22635017335414886, -0.014155127108097076, -0.7256314158439636, 1.0581891536712646, 0.05060353875160217, -1.051126480102539, -0.44774338603019714, -0.07881753146648407, 0.08058002591133118, -0.9152230024337769, 1.9434330463409424, -0.2076539695262909, 0.30671316385269165, -0.023894116282463074, 0.3768833577632904, -0.030310742557048798, -0.7371702790260315, -0.4484955966472626, 0.16100263595581055, 0.481261670589447, 0.8736168146133423, 0.056345291435718536, -0.6299296617507935, -0.44285017251968384, -0.16209997236728668, 0.1173616498708725, 0.5083993077278137, -0.8340986967086792, -0.38424259424209595, 0.5753464698791504, -0.46005529165267944, -0.4939310848712921, -0.22633473575115204, 0.537966787815094, -0.5288545489311218, 0.05692555010318756, 0.07872188836336136, -0.6551718711853027, 0.5236997604370117, -0.1317644715309143, -0.7676027417182922, -0.2502173185348511, -0.23933516442775726, -1.5684278011322021, 0.792007327079773, 0.049810443073511124, 0.37201955914497375, -0.46355828642845154, 0.043707579374313354, -0.9436321258544922, 0.024102304130792618, -0.7869693636894226, 0.2545802891254425, 0.4495335817337036, 0.338325172662735, -0.42868995666503906, -0.009521286934614182, -0.3408825397491455, -0.6919023394584656, -0.2521458566188812, 0.01749826967716217, 0.2095036804676056, -0.050271086394786835, -0.269070029258728, 0.7054762840270996, 0.2533683776855469, 0.6379229426383972, 0.5700728893280029, 0.18677954375743866, -0.7714719176292419, -0.7704558968544006, -0.04789595305919647, -0.7913178205490112, 0.6112634539604187, -0.16269174218177795, -1.0204274654388428, -0.12179623544216156, 0.05304441601037979, 0.512310266494751, 0.4262288510799408, -0.6409481763839722, -0.41356703639030457, 0.4138454794883728, -0.2197437882423401, 0.28739550709724426, -0.74042809009552, 0.7414835691452026, 0.21154287457466125, 0.40467438101768494, -0.6470701694488525, 0.2138689160346985, -0.6184300780296326, -0.7209699749946594, 0.032966747879981995, -0.13679450750350952, 0.23394827544689178, -0.6310045123100281, 0.5572221279144287, 0.5959806442260742, 0.2886800467967987, -0.18975447118282318, 0.6753356456756592, -0.08486064523458481, 0.1521369218826294, -0.41986653208732605, 1.1026532649993896, -0.3909028470516205, 0.2322853058576584, -0.17543472349643707, -0.5576440095901489, -0.59613436460495, -0.08893007040023804, 0.013010052032768726, -0.1889776885509491, 0.06711188703775406, 0.889026403427124, -0.3634085953235626, 0.3607316315174103, 1.0815938711166382, 1.0912351608276367, -0.5280225276947021, 0.8133313059806824, 0.5115299820899963, 0.14082595705986023, -0.14003904163837433, -0.19068102538585663, 0.4030381441116333, 0.13903144001960754, -0.9345289468765259, -0.4374609589576721, 1.0570123195648193, 0.29990851879119873, 0.4526204466819763, -0.15070798993110657, 0.24376939237117767, -0.35059258341789246, -0.19535011053085327, 0.09438017755746841, 1.4868978261947632, 0.24869413673877716, 0.44479313492774963, -1.5544873476028442, -0.7960094213485718, 0.03067566454410553, 0.5662633180618286, 0.71293705701828, 1.032731533050537, -0.799247145652771, -0.8917602896690369, -0.02486754208803177, 0.03013569861650467, -0.3596249520778656, -0.19398266077041626, -0.3485470414161682, 0.32497772574424744, 0.09890668094158173, -0.8852861523628235, 0.06097833067178726, 0.2073344886302948, -0.2165139764547348, 0.42147842049598694, -0.6014145612716675, 1.0252426862716675, -0.35071200132369995, -0.6842597723007202, -0.010830409824848175, -0.32336029410362244, 0.3359052240848541, 0.24567142128944397, 1.0750489234924316, 0.37843966484069824, 0.5877244472503662, -0.2392713725566864, -0.1781485229730606, -0.42081499099731445, 1.4568489789962769, 0.8052625060081482, 0.10234847664833069, 0.10352099686861038, 0.7333860993385315, -0.12672950327396393, -0.4447270631790161, -0.055311691015958786, -0.5068285465240479, -0.9536945819854736], "sim": 0.448087602853775, "freq": 96.0, "x_tsne": -0.41261667013168335, "y_tsne": -27.06133270263672}, {"lex": "poisons", "type": "nb", "subreddit": "conspiracy", "vec": [1.8456573486328125, -0.7379056811332703, -1.1664855480194092, -1.8532763719558716, 0.7998267412185669, 0.7764649391174316, 0.652226448059082, 0.6922348737716675, 0.29586392641067505, -0.8050850033760071, -1.3889104127883911, -2.028008460998535, 1.0628571510314941, -2.0592598915100098, 0.2130831480026245, 0.26540860533714294, 0.8363077044487, -1.1261637210845947, 0.8835781812667847, -0.4294643998146057, 0.028093602508306503, -0.11893998086452484, -1.360501766204834, -1.4836716651916504, 2.722801685333252, 1.1850404739379883, 1.1011208295822144, 0.921302080154419, -1.6398754119873047, 0.5375970602035522, 1.0890110731124878, 0.6185669898986816, -0.8312345743179321, -1.2029330730438232, 1.098229169845581, 1.1510508060455322, 0.23910120129585266, 1.21554434299469, -1.6122448444366455, 0.28026503324508667, -1.3144680261611938, 0.5578619837760925, -0.7462279200553894, 0.751888632774353, 0.24668429791927338, 0.8972206711769104, 1.9113086462020874, 0.6744851469993591, 2.9168457984924316, -1.3443950414657593, 0.56888347864151, 0.19514545798301697, -0.1906464695930481, 0.5502976179122925, -0.9492194056510925, -0.9131100177764893, -0.1775982528924942, -0.7225049734115601, 0.15082357823848724, -0.49376049637794495, -0.7015489339828491, -1.36409592628479, 0.7984721660614014, -0.4080958664417267, 0.34992653131484985, 0.1059870645403862, 1.9703232049942017, 0.7023928761482239, 1.0065521001815796, -0.2249382585287094, 2.1180548667907715, -0.43300682306289673, 1.080744743347168, -0.5762573480606079, 1.2887239456176758, -0.47410574555397034, -0.7115356922149658, 1.4482126235961914, -1.4247922897338867, -0.08198968321084976, 0.7385714054107666, -0.5083291530609131, -0.6996539831161499, 0.897700309753418, -1.0394010543823242, 0.976166844367981, -0.5576424598693848, -0.019104542210698128, -0.40739622712135315, 1.4624969959259033, -0.024504154920578003, -2.0079023838043213, 0.9739553332328796, 0.3213614821434021, -2.145557165145874, 0.5775249600410461, 2.135514736175537, -0.2694140374660492, 0.11351017653942108, -1.0419273376464844, -0.06272020936012268, -0.5524782538414001, -1.6649667024612427, -0.29006388783454895, -1.9359028339385986, -1.247349500656128, 0.11167927086353302, 0.05465927720069885, -0.8268078565597534, 1.1356936693191528, -1.6187366247177124, -1.2405778169631958, -0.36338022351264954, -0.6416342854499817, -2.12823224067688, 0.7745451331138611, 0.8243111968040466, -1.7126127481460571, -0.18864279985427856, 0.17466416954994202, 0.49359431862831116, 0.9383149743080139, -1.3623491525650024, -0.0005043325945734978, 0.767893373966217, 1.759871244430542, 0.05727844685316086, -1.7112427949905396, 1.3883136510849, 0.19692914187908173, -0.35291749238967896, 1.8767677545547485, -2.1297354698181152, 0.6277583241462708, -1.180121660232544, -0.06420278549194336, -1.5725476741790771, 0.24478675425052643, 1.7167472839355469, 1.2422082424163818, 0.6099873781204224, -1.0719785690307617, 0.4683660864830017, -0.20683756470680237, 0.013574032112956047, -1.3077906370162964, -0.8187916874885559, -0.9624167084693909, -0.3712272346019745, -0.9725000858306885, -0.11471018195152283, 0.008937809616327286, 0.4788241684436798, -0.48066845536231995, -0.12503007054328918, -1.3725006580352783, -0.8515705466270447, 0.015026040375232697, -0.18406793475151062, -1.2706478834152222, 0.309824675321579, -1.1762694120407104, -1.1785253286361694, 0.018809856846928596, -0.5943701267242432, -0.01309562660753727, -1.1119415760040283, 0.3455924987792969, -2.234710216522217, 1.6275655031204224, 0.3754231929779053, 1.422584056854248, -0.21081465482711792, 0.7914741039276123, 0.8470804691314697, -0.08375563472509384, -0.967150092124939, 0.5490017533302307, -0.49882593750953674, -1.6055318117141724, 0.4865395128726959, 1.2810277938842773, 0.46995773911476135, -0.017407409846782684, 0.2669752538204193, -0.7943630814552307, 1.5852307081222534, 0.34000036120414734, 0.2541738450527191, 0.8925559520721436, -0.7127994298934937, -0.2890554964542389, 0.9022769927978516, 0.1705433428287506, -0.5792784094810486, -0.5256496071815491, -0.6485276222229004, 0.9840449690818787, 1.0882883071899414, 0.008994920179247856, -2.0459117889404297, 1.4118555784225464, 0.4162288010120392, 1.110684871673584, -0.7800908088684082, -1.620996117591858, 0.3818719983100891, 0.07174106687307358, 0.6680507063865662, 1.826886534690857, -0.5118829011917114, -1.3961336612701416, -1.3037362098693848, -0.009012334048748016, 0.17231151461601257, -0.8169622421264648, -0.2978805899620056, -0.048363812267780304, -0.6178123354911804, 0.47596824169158936, -0.6648555397987366, 0.9834045767784119, -0.0824761688709259, 1.9925992488861084, -0.310085654258728, 0.30418047308921814, 0.6861838102340698, -1.0562771558761597, -0.3412818908691406, 0.7386371493339539, 1.3347350358963013, 0.5303180813789368, -1.1714458465576172, -0.518208384513855, 1.6841334104537964, -0.3587798774242401, 0.671313464641571, 1.4402152299880981, 1.7770929336547852, 0.23732194304466248, -0.7007506489753723, 0.30073314905166626, -0.06029939651489258, -0.7286421060562134, 0.398077130317688, -0.3692499101161957, 0.21226994693279266, 1.4495185613632202, 1.8412684202194214, 0.8598901629447937, 0.260653555393219, -1.1578190326690674, 1.6952437162399292, -0.07882145792245865, 2.371218681335449, 0.4965927004814148, -2.549011707305908, -1.2893729209899902, -2.319146156311035, 0.7390391826629639, 0.9611378908157349, 0.2337687760591507, -0.5140826106071472, -1.6178702116012573, 0.2983520030975342, 0.3052258789539337, 0.48569148778915405, 0.4339037835597992, 1.260467290878296, 0.016866255551576614, -0.499690979719162, -0.08951105177402496, -1.624788761138916, -0.6185007691383362, 1.4284712076187134, 0.12326919287443161, 0.21886669099330902, 0.3024680018424988, 0.3817199468612671, -0.19426578283309937, -0.9149084091186523, 1.0171120166778564, -0.3440190255641937, 0.5811497569084167, 1.237112045288086, -0.5690525770187378, 0.7997338771820068, -1.5015313625335693, -1.7411460876464844, 1.0418305397033691, 0.23323023319244385, -0.6740797162055969, 1.4538438320159912, 0.41846317052841187, 1.001887559890747, 1.1317205429077148, -0.1097748875617981, 0.9366516470909119, -0.6250194311141968, -0.36241596937179565], "sim": 0.435996413230896, "freq": 234.0, "x_tsne": -27.280765533447266, "y_tsne": -0.5087164044380188}, {"lex": "newborns", "type": "nb", "subreddit": "conspiracy", "vec": [-0.8840270042419434, 1.0971319675445557, 0.5793179869651794, 0.3143883943557739, 0.043126024305820465, 0.6913586854934692, -0.27362093329429626, -0.38182732462882996, -0.24050864577293396, -0.13161204755306244, 0.14830242097377777, -0.013074472546577454, 0.45110052824020386, -1.0199456214904785, 1.2327998876571655, 0.7251764535903931, -0.7392553687095642, -0.6827919483184814, 1.342434048652649, 0.7193138599395752, 0.3785589933395386, 0.5912365317344666, -0.11165468394756317, -0.5184781551361084, 1.6487529277801514, -0.5463324189186096, -0.12248788774013519, 0.23297572135925293, -0.5888671875, -1.8201820850372314, -0.22027063369750977, -1.1919584274291992, 0.5959150791168213, -0.5211437344551086, 0.9313750267028809, 0.9375011920928955, -0.17759227752685547, -1.5837304592132568, -0.3391377925872803, -0.24314256012439728, -0.16924023628234863, -0.29564735293388367, -1.4989680051803589, 1.2246770858764648, -0.9480026960372925, 0.16603362560272217, 0.2362525761127472, 0.21700575947761536, 2.2050974369049072, 0.23269890248775482, 0.7313498854637146, -0.08159895986318588, -0.4140631854534149, 0.8594064116477966, -1.097961187362671, 0.028553495183587074, 0.7067453861236572, -0.5317099690437317, 1.166680097579956, 0.5250646471977234, 0.16515079140663147, 0.21154606342315674, -0.154319167137146, 1.1832295656204224, -0.08549715578556061, -0.23463159799575806, -0.32656601071357727, 1.3066282272338867, -0.3443085849285126, 1.4821736812591553, 0.368316650390625, 0.6214372515678406, -0.8091594576835632, 0.4500387907028198, -0.597260057926178, -0.48386648297309875, -0.06293404847383499, -0.6112232804298401, -1.7143980264663696, -0.9873044490814209, 0.17643418908119202, 0.16103366017341614, 0.46519386768341064, 1.49409019947052, 0.24838963150978088, -0.42156296968460083, -0.565769374370575, -0.5582993626594543, -0.02701510675251484, -0.27311089634895325, 0.29461538791656494, -0.38744574785232544, 1.7291350364685059, 0.7044184803962708, -1.8498414754867554, -0.08982044458389282, -1.3479520082473755, 0.16134251654148102, 0.04103279113769531, 1.027197241783142, 0.5174268484115601, 0.5410895347595215, 0.020931798964738846, -0.566827118396759, -2.219914197921753, 0.7875264883041382, -1.0240662097930908, -0.8690369129180908, 0.46873173117637634, -0.6377784013748169, -0.9393877387046814, -1.8314908742904663, -0.36221155524253845, -0.9600902199745178, -0.6083559989929199, 1.127821922302246, -0.3349018692970276, 0.17620408535003662, 0.18337732553482056, 0.25488775968551636, -0.6055472493171692, -1.7171859741210938, -0.5852148532867432, -0.05136123672127724, -0.3387387990951538, 0.41792505979537964, 1.2159550189971924, -0.3370136022567749, -0.09552731364965439, -0.34535154700279236, -0.2998673915863037, 0.46115773916244507, 0.16764923930168152, 0.7138849496841431, 0.37932300567626953, -0.5787895917892456, -0.7274123430252075, -2.64254093170166, 0.9410784840583801, 0.8260879516601562, -0.06443145871162415, 0.22141028940677643, 0.3106692433357239, 0.10558169335126877, -0.603943943977356, -0.4620913863182068, -1.8487588167190552, -0.5690656304359436, 1.3241277933120728, -1.0313401222229004, 1.0277596712112427, -1.4338363409042358, -0.5004294514656067, 0.006309633143246174, -1.2832555770874023, -0.18386821448802948, -0.724775493144989, -0.5316921472549438, -1.1104960441589355, 0.8682714104652405, 0.05998985469341278, 1.3545331954956055, -1.6521635055541992, 0.44100067019462585, -0.03267308324575424, 0.26939839124679565, -0.2926011085510254, -0.34303706884384155, 0.22703108191490173, 0.5501055121421814, -0.11750135570764542, 0.5312371850013733, 0.2318093180656433, -0.1848343312740326, 0.5760499835014343, 0.016713064163923264, 0.18311050534248352, -0.9012951254844666, -1.4937679767608643, -0.5745278000831604, -0.48813021183013916, 1.1274570226669312, -0.16590313613414764, 0.8321143388748169, -1.1290669441223145, -0.41735759377479553, -0.4443398714065552, 0.6934199333190918, 1.7857705354690552, -0.02339453622698784, -0.4853854477405548, -0.1291159987449646, 0.027542170137166977, 0.8303616046905518, -0.8465810418128967, -1.7110947370529175, -0.8631919026374817, -0.7117987871170044, 0.5933355093002319, -0.7277672290802002, -1.9516421556472778, 0.12390289455652237, -0.24852298200130463, 1.099053144454956, -1.309697151184082, -1.1053857803344727, -1.0166990756988525, -0.7548739910125732, 0.5680559277534485, 0.5015149116516113, 1.1297379732131958, -1.9386497735977173, 1.0899033546447754, 0.8125448822975159, -0.3134572505950928, 0.8653454184532166, -0.9749889373779297, -0.701632559299469, -0.3810441493988037, -0.08187346905469894, 0.1951589584350586, 0.15155327320098877, 1.2978248596191406, -0.3372403085231781, -1.1417036056518555, -0.4815574586391449, 1.063283085823059, -0.471406489610672, 0.25485867261886597, -0.11032809317111969, 1.2325079441070557, -0.8578096628189087, -0.28659963607788086, 0.8905576467514038, 1.7568060159683228, 0.8327474594116211, 1.1889371871948242, 0.11639536917209625, -0.376922607421875, 0.026797158643603325, -0.036844778805971146, 0.4430513381958008, 0.3527592122554779, -1.3754059076309204, -0.177579864859581, -0.30019664764404297, 0.22416886687278748, 1.126757264137268, 0.5349987149238586, 0.5880867838859558, -0.7443050742149353, -1.7101666927337646, 0.7203871011734009, 1.4618831872940063, 0.9444525837898254, 1.3510531187057495, -0.2718958258628845, -0.010695749893784523, 0.21591222286224365, 0.2476024627685547, 0.3548019230365753, 1.0921884775161743, 0.1402367204427719, -1.1849743127822876, -0.2597567141056061, -0.33268314599990845, -0.3165184259414673, 0.5246657133102417, 0.011179567314684391, 1.2019859552383423, 0.15377570688724518, -0.8788779973983765, -1.6227998733520508, -0.25292491912841797, 0.49547871947288513, -0.1690744161605835, -0.08491078019142151, 1.2179862260818481, -0.9355930685997009, 0.7167173624038696, -0.029951803386211395, -0.7943121194839478, 0.02452789433300495, 0.6816021203994751, 0.9541264772415161, -0.7028255462646484, -0.1737901270389557, 0.03653745353221893, -0.29643747210502625, -0.01891537383198738, 1.2158223390579224, 0.6439139246940613, 0.03524748608469963, 0.1256997287273407, 0.6808295249938965, 0.09487766772508621, 0.20389865338802338, 0.4869885742664337, 0.39778026938438416, -0.493625283241272], "sim": 0.43558749556541443, "freq": 243.0, "x_tsne": 26.790332794189453, "y_tsne": 26.527603149414062}, {"lex": "hcg", "type": "nb", "subreddit": "conspiracy", "vec": [-1.4447909593582153, 0.8838303685188293, 1.3903131484985352, -0.9409151673316956, 1.010615348815918, 0.22664186358451843, -1.5661762952804565, 0.08021610230207443, 0.22751356661319733, 1.0341098308563232, -0.15617568790912628, -1.0020458698272705, 2.3801040649414062, -1.6201289892196655, 1.4944664239883423, -0.3864341378211975, 0.7362171411514282, -0.17719070613384247, -0.021599093452095985, 0.5639790296554565, -0.12430360913276672, 0.6677320003509521, -0.7261606454849243, 1.0689560174942017, 1.2693262100219727, -0.565099835395813, -0.08612769097089767, 0.4275587499141693, -0.9320868849754333, -0.4045637845993042, -0.022456355392932892, -2.209815740585327, -0.060127027332782745, 0.018589448183774948, 1.269273042678833, 1.0170161724090576, -0.2239316999912262, -0.2078622728586197, -1.5111327171325684, 1.2835140228271484, -0.3437575399875641, -0.07323334366083145, 0.5533804297447205, 0.4881572425365448, 0.7768972516059875, 0.7550331950187683, 1.3794163465499878, 1.6226398944854736, 0.7421153783798218, -0.7883448004722595, 1.007914423942566, 0.7344744801521301, -1.2584006786346436, -0.08523110300302505, -2.0544142723083496, -0.4823683202266693, 0.4962127208709717, -1.4838883876800537, 1.6433563232421875, -0.7057146430015564, 0.26275646686553955, 1.6993516683578491, -0.8148980140686035, 1.4241915941238403, -0.1406254768371582, -0.8673890233039856, 0.02329362742602825, 0.2809480130672455, -0.7387669086456299, 2.490229368209839, 1.3631184101104736, -0.4820161461830139, -0.4041880667209625, 0.9687137603759766, 0.5520628690719604, -0.9895632863044739, -0.7581940293312073, -0.16263163089752197, -0.731997013092041, -0.009676769375801086, -1.687753438949585, 1.2397925853729248, 0.2347029447555542, 1.5375912189483643, -1.7985012531280518, -0.19546660780906677, -0.76151442527771, -1.335476040840149, 1.5662610530853271, 1.0695170164108276, 0.48098722100257874, -1.4061514139175415, 0.9168853759765625, -0.03596033900976181, -0.7733940482139587, 0.9705390930175781, 0.09601838141679764, 0.2240792065858841, 2.2868196964263916, 0.30373451113700867, 0.3612444996833801, -0.7918286323547363, -0.021410992369055748, -0.11846773326396942, -2.544419050216675, -0.31232550740242004, 0.19924825429916382, -1.2217934131622314, 0.0702492892742157, 0.5965892672538757, -1.161242127418518, -2.2402312755584717, 0.3510454297065735, 1.0647350549697876, -0.4102681279182434, -0.4822938144207001, 1.1134661436080933, 0.09155235439538956, 0.7186185121536255, -0.6911199688911438, -0.8909004926681519, 0.9454476237297058, 0.46199941635131836, 0.7894440293312073, 0.19774673879146576, 1.5342097282409668, -0.9885556697845459, -0.22896474599838257, -1.3212978839874268, 0.906450629234314, -0.09750238060951233, 0.6715675592422485, -0.8838543891906738, 1.1166303157806396, 0.5184509754180908, -0.9932382702827454, -2.121241331100464, -0.8567988276481628, -0.19202721118927002, -3.034398317337036, -0.7058854103088379, -1.6319103240966797, -1.3666880130767822, -0.6375862956047058, -0.13404951989650726, -1.9035611152648926, -1.8216801881790161, 0.13568809628486633, 1.0354013442993164, 0.737677812576294, 0.8357357382774353, -0.5017136931419373, -0.8438429236412048, 0.2846013009548187, -1.4766483306884766, 0.46380776166915894, -1.6845786571502686, -0.6948286294937134, -1.2186625003814697, 0.8270837664604187, -1.1190855503082275, 0.5272196531295776, -0.5862025618553162, -0.10577122867107391, -1.1552314758300781, -1.9754141569137573, 0.7256224155426025, 0.5457593202590942, -0.10772503167390823, 0.7988923788070679, 0.6048163771629333, -0.381148099899292, -1.052459955215454, -0.20098146796226501, -0.4482126235961914, -0.4801355302333832, 0.34805911779403687, -0.25771525502204895, -1.0345486402511597, 0.1109519824385643, -0.34407126903533936, 1.0648235082626343, -0.805766224861145, 1.2232869863510132, -0.4637654721736908, -0.2777232527732849, 0.683910071849823, 0.310230016708374, 1.412500262260437, -0.4092426896095276, -2.3986308574676514, -0.17998284101486206, -0.38575515151023865, -0.9905678033828735, 1.3190408945083618, 0.5212118029594421, 2.124037981033325, 0.5434911251068115, -0.46465200185775757, 0.4578935205936432, -2.2041680812835693, 0.18867994844913483, -1.0244172811508179, 0.06547705084085464, -0.510007381439209, 0.161005899310112, -1.0086534023284912, -0.3669905662536621, -0.28753137588500977, 1.8233957290649414, -1.1103249788284302, -1.6253637075424194, -0.46664249897003174, 0.3218264877796173, 1.8674192428588867, -0.4416709244251251, -0.2666625678539276, 0.4511528015136719, -0.5253779888153076, -1.1522667407989502, 0.8673100471496582, 1.4067398309707642, 0.6870047450065613, 0.9078407287597656, -2.1081323623657227, 0.7688624858856201, -0.27603426575660706, 0.2609699070453644, -0.9301166534423828, 0.9626478552818298, 1.258723258972168, -1.2309449911117554, -0.01827091909945011, 0.5908534526824951, 0.5754324197769165, -1.0113942623138428, 1.9667208194732666, 2.075967788696289, -1.0720734596252441, -1.4923986196517944, 0.9884052276611328, 0.4411452114582062, 0.08887799084186554, -1.9188454151153564, -0.1125490590929985, 0.2634313404560089, 0.8542671203613281, 0.5289900302886963, 0.6884970664978027, -0.5075621604919434, 1.5160272121429443, -0.07331853359937668, 1.4895230531692505, 1.3898106813430786, -0.332180917263031, -0.2855318784713745, -0.7277514934539795, 0.36074721813201904, -0.9309669137001038, -1.274993896484375, 0.08228059858083725, 1.6585586071014404, 0.12513959407806396, -0.7646134495735168, 1.139067530632019, -0.10891926288604736, -0.41448912024497986, -0.23461151123046875, 0.4304102063179016, -1.176305890083313, -0.5201092958450317, -2.835639715194702, -3.264065980911255, 0.5190638899803162, 0.8246018886566162, 2.204094886779785, -0.9221557378768921, 3.431849241256714, 1.1794699430465698, 0.041169799864292145, 0.6474217772483826, 0.8732454180717468, -0.031746167689561844, -0.24491417407989502, 1.3482633829116821, -0.6875531673431396, 1.363329529762268, -0.44566383957862854, -1.501064419746399, -0.40727949142456055, 1.9727003574371338, -0.024408742785453796, 0.8949866890907288, 1.1457240581512451, 0.6570886373519897, -1.168870449066162, 0.14445777237415314, 1.3625695705413818, 0.7837332487106323, 0.2728818655014038], "sim": 0.4289657771587372, "freq": 114.0, "x_tsne": -16.534988403320312, "y_tsne": -5.873934745788574}, {"lex": "ultrasounds", "type": "nb", "subreddit": "conspiracy", "vec": [-0.5233147740364075, 0.025302473455667496, -0.3011532127857208, -0.06735667586326599, -0.1678883582353592, 0.12913434207439423, -0.15219467878341675, 0.4303397238254547, 0.7354274392127991, 0.0663067027926445, 0.882738471031189, 0.32052674889564514, 0.2783423662185669, -0.2818746268749237, 0.3183431029319763, 0.13778764009475708, 0.30156391859054565, 0.11696182936429977, 0.2717054784297943, 0.12661808729171753, -0.11752568185329437, 0.9780358076095581, 0.6495800018310547, -0.47371894121170044, 0.4658640921115875, -0.03563566133379936, -0.37881040573120117, 0.6362451910972595, 0.03946465998888016, 0.22872188687324524, -0.06033675745129585, -0.10104050487279892, 0.6970945596694946, -0.20031630992889404, 0.30108776688575745, 0.7759974598884583, 0.0425340011715889, 0.19006919860839844, -0.07030603289604187, 0.0025702379643917084, 0.23414725065231323, -0.12060093879699707, -0.502910315990448, 0.5927092432975769, 0.3490537703037262, -0.38475990295410156, 0.23277701437473297, 0.31238555908203125, 1.7325634956359863, -0.2383124977350235, 0.27184757590293884, 0.10697685182094574, 0.020861748605966568, 0.08248549699783325, -0.7167454361915588, -0.12769384682178497, 0.4808579683303833, 0.018318597227334976, -0.002414085902273655, -0.6419140100479126, -0.13104510307312012, 0.18727341294288635, -0.6449583172798157, 0.6082921028137207, 0.19263532757759094, -0.03210831806063652, 0.10531272739171982, -0.06289123743772507, -0.442303329706192, 1.4760535955429077, 0.41238415241241455, 0.9481475949287415, 0.48446303606033325, -0.18188539147377014, -0.037812739610672, -0.5718149542808533, -0.03023199550807476, -0.23051351308822632, 0.2884114980697632, -0.7605927586555481, 0.5619127750396729, -0.3080666959285736, -0.2091653048992157, 0.07292843610048294, -0.4009777903556824, -0.1343991905450821, -0.6585831642150879, -0.755561888217926, 0.07465922087430954, -0.024739116430282593, -1.084402322769165, -0.0581837072968483, 0.7675327658653259, 1.0471153259277344, -0.08686953037977219, -0.378120094537735, 0.26666274666786194, 0.3726446330547333, 0.2181163728237152, 0.553634524345398, -0.09606581181287766, 0.08098238706588745, 0.13748233020305634, -0.521605372428894, -0.33469998836517334, -0.06329366564750671, 0.6577664613723755, 0.14264194667339325, 0.25337037444114685, 0.3810558319091797, -0.3486492335796356, -0.722875714302063, -0.06611933559179306, 0.08654849976301193, -0.3885086476802826, 0.12755481898784637, -0.7718276381492615, -0.1641264110803604, 0.10630371421575546, 0.12510238587856293, -0.15209515392780304, -0.12654227018356323, 1.0961973667144775, -0.06578704714775085, 0.8507600426673889, 0.4354574978351593, 0.129744291305542, 0.28184327483177185, -0.111277274787426, 0.2413351982831955, -0.27393704652786255, -0.6843670606613159, -0.002894713543355465, -0.3305422067642212, 1.1133347749710083, -0.16857990622520447, -0.6136072278022766, -1.139079213142395, 0.3105004131793976, 0.027625612914562225, 0.13886430859565735, 0.14464035630226135, 0.0277651809155941, -0.06634283065795898, -0.4304026663303375, 0.1627316027879715, -0.05618535727262497, -0.18627358973026276, 0.27724015712738037, -0.41921746730804443, 0.06324511021375656, 0.14985516667366028, 0.33606648445129395, 0.27056968212127686, -0.0061180200427770615, -0.3710077106952667, -0.49228477478027344, -0.5204549431800842, -0.6628084778785706, -0.08084306120872498, -0.09273587167263031, 0.9022499322891235, -0.45112794637680054, -0.15206339955329895, -0.5660635232925415, 0.29116567969322205, -0.019619934260845184, 0.0325554795563221, 0.06045479699969292, 0.03685067966580391, 0.8263572454452515, -0.6218879222869873, 0.12429150938987732, 0.020548885688185692, 0.18473801016807556, -0.21959243714809418, -0.084161177277565, -0.001779417973011732, -0.21455402672290802, -0.22978830337524414, 0.35389649868011475, 0.32923102378845215, -0.5345075726509094, -0.4171694815158844, 0.17612366378307343, 0.6653363704681396, 0.09101548790931702, 0.2052782028913498, 0.6545662879943848, -0.3270624577999115, -0.28120332956314087, -0.3169299364089966, -0.6153534054756165, -0.21872368454933167, -0.45144006609916687, -0.176267609000206, -0.8139125108718872, 0.4049935042858124, 0.25816795229911804, -0.02621234953403473, 0.17877215147018433, 0.3955889046192169, 0.16351324319839478, 0.8503482937812805, -0.8884071707725525, -0.08483187854290009, -0.684421956539154, -0.3747715353965759, 0.1259693205356598, -0.5571819543838501, -0.9158643484115601, -0.43635326623916626, 0.15139472484588623, -0.43243613839149475, -0.13137975335121155, 0.12777195870876312, -0.5090230107307434, 0.630524218082428, 0.6372979283332825, -0.14220713078975677, -0.1258300393819809, 0.2180316299200058, -0.0970134437084198, 0.5681017637252808, -0.4652605652809143, -0.1635921150445938, 0.9011996388435364, -0.1149195209145546, -0.26172399520874023, -0.6251014471054077, 0.49149760603904724, -0.9712668061256409, -0.7358623147010803, 0.1799561083316803, 0.9629572033882141, 0.6467975974082947, 0.27996546030044556, 0.740498960018158, 0.2563886046409607, 0.08625302463769913, 0.03581324964761734, -0.46013057231903076, -0.24394448101520538, -0.626500129699707, -0.3650563359260559, -0.22987036406993866, -0.36791205406188965, -0.464218407869339, 0.4464460611343384, 0.8805834650993347, 0.39893394708633423, 0.05965268984436989, 1.227787733078003, 0.1387138068675995, 0.2821539044380188, 0.5089971423149109, -0.8184438943862915, -0.05430576950311661, 0.478916734457016, -0.5133066177368164, -0.29499053955078125, 0.3627759516239166, 0.08169455826282501, 0.07301364094018936, -0.24075354635715485, -0.7456436157226562, -0.4716985821723938, -0.38244959712028503, 0.13628675043582916, -0.2050398886203766, 0.05862364172935486, -0.0741095170378685, -0.5319291949272156, -0.5705463290214539, 0.26010918617248535, 0.09192381799221039, -0.504554808139801, 0.4946083128452301, 0.05826837196946144, 0.16904792189598083, -0.2664965093135834, -0.5838775634765625, -0.7206991910934448, -0.3520848751068115, 0.770333468914032, -0.8099080324172974, 0.25011447072029114, -0.15959517657756805, 0.006018362008035183, 0.798650324344635, 0.788059413433075, 0.7064951658248901, -0.025838486850261688, -0.3747956454753876, 0.6435778141021729, -0.5467187166213989, -0.15403546392917633, 0.11474771052598953, -0.9389929175376892, -0.14800231158733368], "sim": 0.4281875193119049, "freq": 54.0, "x_tsne": 6.7973833084106445, "y_tsne": -11.670763969421387}, {"lex": "vaxx", "type": "nb", "subreddit": "conspiracy", "vec": [-0.1096653938293457, 1.4282726049423218, 0.19846399128437042, 1.1065369844436646, -2.2007715702056885, -1.274547815322876, -0.23547793924808502, 0.24444565176963806, -0.20633268356323242, 1.8032124042510986, 0.5869289636611938, 1.1317180395126343, 1.0769717693328857, 1.2739354372024536, 2.7702436447143555, 0.8379085063934326, -0.7245728373527527, 0.30936112999916077, -1.0565017461776733, 2.79630708694458, 1.03059983253479, -0.42999231815338135, -0.5074344873428345, 1.7345472574234009, 0.48236674070358276, 0.4682675898075104, 1.2434338331222534, 0.849699854850769, 1.1733723878860474, 0.6636868715286255, -0.8001540303230286, -1.8169974088668823, 0.6442651152610779, -2.230559825897217, -2.2118988037109375, -0.43027839064598083, 1.5434938669204712, -0.6162527799606323, -0.27617910504341125, 1.1324374675750732, -0.3409762978553772, -0.3898419141769409, -0.22675123810768127, 0.2147134393453598, 0.32084670662879944, 0.22010457515716553, 0.6998757123947144, -0.2221965789794922, 1.8207218647003174, -0.8476724624633789, -0.29601308703422546, -0.8175147175788879, -0.5056870579719543, -1.5029215812683105, -2.6388752460479736, -1.3586349487304688, 0.8101659417152405, -0.940735399723053, 0.5453988909721375, -2.1312508583068848, -0.4045967161655426, -0.5502510666847229, -2.669321060180664, 0.5290042757987976, 2.231639862060547, 0.34634673595428467, -0.18396222591400146, 0.12292712181806564, -0.1720300167798996, 1.5470701456069946, -0.3768827021121979, -0.02491411194205284, 1.443955421447754, 3.4464993476867676, 0.06897816807031631, -2.0295848846435547, 0.12839771807193756, 0.4032333493232727, 0.5869213342666626, -1.1600818634033203, -0.624921977519989, -2.121829032897949, 0.619662880897522, 0.9407115578651428, -1.5596168041229248, -0.025191769003868103, -1.7283052206039429, -1.7847825288772583, -1.2544338703155518, 0.20777371525764465, -0.27418601512908936, 0.33102643489837646, 0.9779417514801025, 1.1715624332427979, -0.5402897000312805, 0.20748886466026306, -0.43574652075767517, -0.5064029693603516, -0.3299672603607178, 0.5962694883346558, 0.7465375661849976, -1.4617470502853394, 0.34755992889404297, -2.076803207397461, -0.854862630367279, -3.4121320247650146, 0.6144363880157471, 0.03246515244245529, 1.057885766029358, 0.9108163118362427, 1.1143615245819092, -3.5391476154327393, 0.06387278437614441, 0.6708843111991882, -0.1730838418006897, -1.262505292892456, 0.30990535020828247, 1.0967752933502197, 1.7617868185043335, -1.8792170286178589, 1.019209861755371, 0.0985802635550499, 0.576539933681488, 0.15648186206817627, 1.6007953882217407, 1.9814881086349487, -0.9331754446029663, -1.2764264345169067, -0.9910817742347717, 1.8155077695846558, 0.44479262828826904, -0.28144094347953796, 0.8150424361228943, -0.5121867060661316, 0.2034500539302826, 0.27758410573005676, -2.390528440475464, 0.2576550245285034, -1.0395987033843994, -0.6794993877410889, 0.803401529788971, 0.36884599924087524, -1.9970424175262451, 1.4837231636047363, 0.5115050077438354, -2.776566505432129, 0.32550716400146484, -0.6632078886032104, 1.2351205348968506, 0.810089111328125, 0.060501981526613235, 0.5647267699241638, -1.5244823694229126, -2.3184115886688232, -0.6890308260917664, 0.4523312747478485, -0.5050311088562012, 0.8090464472770691, -2.118494749069214, 0.8441256880760193, -0.27774739265441895, 1.1979535818099976, -1.5272830724716187, -0.07060413062572479, -2.1060452461242676, -0.5557127594947815, 0.6808802485466003, 1.8532276153564453, -1.1251832246780396, -1.1693068742752075, 1.4006465673446655, -0.6458294987678528, -0.545366644859314, -1.9496227502822876, -0.09969690442085266, -0.6684638261795044, -0.5534160137176514, -1.602826476097107, -2.424215793609619, 0.30077698826789856, -0.9136722683906555, 1.0684170722961426, -1.1630465984344482, 1.0667712688446045, 2.4706568717956543, -2.2418861389160156, 1.2441740036010742, -0.7198252081871033, -0.03508109226822853, 0.04184633493423462, -2.2455334663391113, 0.20515397191047668, -0.7847455739974976, -0.7142409682273865, -1.3743884563446045, 1.231086015701294, -3.329948663711548, -0.7106838822364807, 0.1510905921459198, -0.19046178460121155, -1.704366683959961, 1.8468749523162842, -0.7149872779846191, 0.3257420063018799, -0.3912414014339447, 0.978873610496521, -2.147076368331909, -2.9775829315185547, 0.21366570889949799, -0.5292938947677612, 0.5016430020332336, -0.7315580248832703, 0.04925142228603363, -0.4876449704170227, 1.0892057418823242, -0.8111593723297119, -0.6590679287910461, 1.2122888565063477, -0.5135846138000488, -1.2407459020614624, -0.7914345860481262, 1.0870364904403687, 0.029118278995156288, 0.7221088409423828, 0.22663888335227966, -1.628931999206543, -1.5592312812805176, 2.934572696685791, -0.5573936104774475, 1.5059913396835327, 0.38956743478775024, -1.4009919166564941, 0.3268514573574066, 1.3069636821746826, 1.7671971321105957, -0.48916205763816833, 1.9025578498840332, -0.1911163628101349, -0.4968704581260681, -0.43304088711738586, 1.1665029525756836, 0.6777519583702087, 1.8408292531967163, -0.4916023015975952, 0.9013184905052185, 0.759658694267273, -0.1530313640832901, -0.9567883610725403, -1.4305837154388428, 1.8244659900665283, -0.1843685507774353, 1.571096658706665, -1.0713295936584473, 0.25633901357650757, 1.1944583654403687, -0.014304996468126774, -0.16600656509399414, -0.12845797836780548, -2.9012258052825928, -0.4976387619972229, -0.4901694357395172, 0.8673476576805115, -1.4366704225540161, -1.2516595125198364, -0.281377911567688, -0.204349085688591, -0.6526986956596375, -0.1924205720424652, 0.4457249045372009, -1.9460445642471313, -1.076322078704834, -1.4130644798278809, -2.296895980834961, -0.3324759304523468, -0.3118835389614105, -0.9460495114326477, -0.22896383702754974, 0.19032083451747894, -3.2869691848754883, -0.5502293705940247, 1.2545015811920166, -0.44034498929977417, 1.402650237083435, 0.6529183387756348, 0.1473560929298401, -0.019015943631529808, 1.8653850555419922, 1.520477056503296, -1.75656259059906, 0.9392687678337097, 0.9020498394966125, -0.06394678354263306, 0.5688825845718384, 0.7571704387664795, 1.3249362707138062, 1.3726118803024292, 0.5720983743667603, 0.6066796183586121, 0.8399895429611206, 1.695542812347412], "sim": 0.42759302258491516, "freq": 770.0, "x_tsne": 12.897775650024414, "y_tsne": -29.96660804748535}, {"lex": "concoctions", "type": "nb", "subreddit": "conspiracy", "vec": [1.2144200801849365, -0.4370061457157135, -0.334823340177536, -1.1649147272109985, -0.33692291378974915, 0.8059454560279846, -0.4548654556274414, 0.3350418210029602, -0.297207772731781, -0.26402124762535095, -0.05999172478914261, -0.4197334051132202, 0.8716879487037659, -0.552056610584259, 0.3367358446121216, 0.023511620238423347, 0.11136284470558167, -0.9320929646492004, 0.7819970846176147, -0.4589819014072418, 0.38922157883644104, -0.9350009560585022, 0.07276422530412674, -0.30449461936950684, 1.4307096004486084, 0.4296271502971649, 0.04303797334432602, 0.6678490042686462, -0.172946497797966, -0.016058482229709625, -0.9209238290786743, 0.4979279935359955, -0.3364405632019043, -0.8447697758674622, 0.42093199491500854, 0.183070570230484, 0.01478656381368637, 0.6496749520301819, -0.4813952147960663, -0.07240410149097443, -0.6073400378227234, 1.0580980777740479, 0.5097013711929321, 1.100290298461914, -0.29902493953704834, 0.11493312567472458, -0.2818251848220825, 0.42541539669036865, 0.28632599115371704, -0.2678459882736206, -0.6164326071739197, 0.20298266410827637, -0.21700835227966309, 0.40061628818511963, -0.5862059593200684, 0.3938889503479004, 0.5455106496810913, -0.6114328503608704, 0.30877402424812317, -0.7153161764144897, 0.25488853454589844, 0.2840576171875, -0.03218110278248787, -0.45030084252357483, 0.02016160637140274, 0.35289567708969116, 0.31812289357185364, -0.6480794548988342, 0.3516084551811218, -0.7152724862098694, 0.8113994598388672, 0.204423725605011, -0.5450584292411804, 0.019202325493097305, 0.2354215383529663, -0.1413339078426361, 0.4925486147403717, 0.3450266420841217, -0.6067694425582886, 0.9437248706817627, 0.6182745099067688, -0.16521480679512024, -0.6944341063499451, 0.7863926291465759, -0.8262022137641907, -0.7754756808280945, -0.3580518066883087, 0.37174171209335327, -0.37246084213256836, 1.0622873306274414, -0.08923500776290894, -0.16194456815719604, 0.6959349513053894, -0.06528972089290619, -1.0002214908599854, -0.03574346750974655, 0.269130140542984, 0.4929591715335846, 0.015888147056102753, -0.18792781233787537, 0.08640967309474945, 0.3280820846557617, -0.05073728412389755, -0.5888214111328125, -0.9305616617202759, -0.4475274682044983, 0.2785606384277344, 0.24942602217197418, 0.135598286986351, -0.14430777728557587, 0.5386419296264648, 0.442726731300354, -0.5607171654701233, -0.459710031747818, -0.9701258540153503, 0.1123107373714447, 0.18644870817661285, 0.11126621812582016, 0.11383605003356934, -0.09139946103096008, 0.15637575089931488, -0.5759746432304382, -0.5407700538635254, -0.8742699027061462, 0.29443109035491943, 1.4858050346374512, 0.6030676364898682, -0.7447860240936279, -0.40182894468307495, -0.09735871106386185, -0.26364320516586304, -1.224007248878479, -0.01099134050309658, 0.0625847727060318, 0.04118794947862625, -0.5123421549797058, -0.25142061710357666, 0.10072138905525208, 0.7273418307304382, 0.32986441254615784, 0.20726805925369263, -0.07981330901384354, 0.005760327447205782, -0.19300927221775055, 0.09774266183376312, -0.6654896140098572, -0.3055874705314636, -0.10460758209228516, 0.6019906401634216, -0.17677927017211914, 0.07405829429626465, 0.28683117032051086, 0.17482203245162964, -0.47949686646461487, -0.06369690597057343, -0.377762109041214, 0.32474496960639954, -0.2904280126094818, 0.04726243019104004, 0.41727784276008606, 0.21214289963245392, -0.08181759715080261, 0.33305624127388, -0.6028195023536682, -0.00929179135710001, 0.1440332978963852, -0.4132356643676758, 0.13259048759937286, -0.2230825573205948, 0.7932789325714111, -0.23393389582633972, 0.5633702874183655, -0.5816324949264526, -0.03776492178440094, 0.4967653453350067, -0.4424717128276825, -0.09588209539651871, -0.5269135236740112, 0.011082949116826057, -0.06532921642065048, -0.31518176198005676, 0.7086265087127686, 1.0629370212554932, 0.9488928914070129, 0.007399367168545723, 0.18753200769424438, 0.7786329984664917, 0.05073907598853111, 0.2757208049297333, 0.8303809762001038, -0.3804279863834381, -0.093776635825634, 0.40011024475097656, 0.38522279262542725, -0.5345190167427063, -0.93229740858078, 0.6136337518692017, 0.3697070777416229, -0.03355754166841507, 0.25760596990585327, -1.2871067523956299, -0.07145440578460693, -0.09097593277692795, 1.183979868888855, -0.5643435716629028, 0.20143401622772217, -0.2966727912425995, -0.18735793232917786, 0.16954711079597473, 0.11410849541425705, -0.676903486251831, -0.7049483060836792, -0.11738412827253342, -0.03765277937054634, 0.2917003631591797, -0.6590400338172913, -0.8075810074806213, 0.2762051224708557, -0.16037899255752563, -0.22575612366199493, 0.23161807656288147, 0.9906650185585022, 0.24856114387512207, -0.5576078295707703, 0.4654173254966736, 0.2727991044521332, 0.7675304412841797, 0.8890902996063232, 0.5727664232254028, -0.07855986803770065, 0.015524635091423988, -0.5277546644210815, 0.08163349330425262, -0.5596519112586975, 0.5960171818733215, 0.3051350712776184, 0.6515673995018005, 0.8169450163841248, -0.2705132067203522, 0.1897563338279724, -0.12574812769889832, 0.7000650763511658, -0.31217652559280396, -0.9597645998001099, -0.3697662949562073, -0.22220134735107422, 0.8563203811645508, 0.25366732478141785, 0.8560842275619507, 0.07297080755233765, 0.33328819274902344, -0.6037898659706116, -0.17102301120758057, 0.030339106917381287, 0.769367516040802, 0.25257259607315063, -0.6905057430267334, -0.3937433958053589, 0.0541435144841671, -0.44075748324394226, 0.11331132799386978, 0.1931750625371933, -0.6729505062103271, -2.1061768531799316, 0.43344801664352417, -0.691731870174408, 0.7038871645927429, 0.5032145380973816, -0.23115673661231995, -0.2375326156616211, -0.9450643062591553, 0.023999283090233803, -0.633671760559082, -0.12686020135879517, 0.8699973225593567, 0.35512545704841614, 0.4684543311595917, 0.9366313219070435, 0.2625102996826172, 0.25874823331832886, -0.5959489941596985, 0.12749704718589783, 0.5227353572845459, 0.9137805700302124, 0.14496250450611115, 0.036012303084135056, 0.6951231360435486, -0.419882208108902, -0.24977992475032806, -0.6770145893096924, 0.4914743900299072, 0.11120365560054779, 0.5120121240615845, 0.6255097389221191, 0.6140958070755005, 0.7767810821533203, -0.6161821484565735, 0.400618314743042, -1.1375811100006104, -0.0557769313454628], "sim": 0.42704376578330994, "freq": 35.0, "x_tsne": 13.514286041259766, "y_tsne": -3.6674695014953613}]}}, {"mode": "vega-lite"});
</script>


```python
chart_diffs.save(f'../out/map-sem-space_{lex}_diffs.pdf')
chart_diffs.save(f'../out/map-sem-space_{lex}_diffs.html')
```

### Dimensions of social semantic variation

<a id='sem-axis'></a>

The following section presents the plots for Figure 2.

```python
lexs = [ 'corona', 'rona', 'moderna', 'sars', 'spreader', 'maskless', 'distancing', 'quarantines', 'pandemic', 'science', 'research', 'masks', 'lockdowns', 'vaccines' ]
```

#### _good_ vs _bad_

```python
pole_words = ['good', 'bad']
```

```python
proj_sims = get_axis_sims(lexs, models, pole_words, k=10)
```

```python
proj_sims = aggregate_proj_sims(proj_sims)
```

```python
proj_sims_chart = plot_sem_axis(proj_sims, models)
proj_sims_chart
```





<div id="altair-viz-f39cf6bccd95453091c0258a0171eb91"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-f39cf6bccd95453091c0258a0171eb91") {
      outputDiv = document.getElementById("altair-viz-f39cf6bccd95453091c0258a0171eb91");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": {"type": "line", "color": "red", "point": {"color": "red"}}, "encoding": {"x": {"field": "conspiracy", "title": "r/conspiracy: red", "type": "quantitative"}, "y": {"field": "lex", "sort": {"field": "SimDiff"}, "title": "", "type": "nominal"}}}, {"mark": {"type": "line", "color": "blue", "point": {"color": "blue"}}, "encoding": {"x": {"field": "Coronavirus", "title": "r/Coronavirus: blue", "type": "quantitative"}, "y": {"field": "lex", "sort": {"field": "SimDiff"}, "title": "", "type": "nominal"}}}], "data": {"name": "data-244f2029e4181be624a777e22fbcc6d3"}, "title": "", "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-244f2029e4181be624a777e22fbcc6d3": [{"lex": "masks", "Coronavirus": 0.07336952537298203, "conspiracy": -0.07284242659807205, "SimDiff": 0.14621195197105408}, {"lex": "vaccines", "Coronavirus": 0.08083311468362808, "conspiracy": -0.03164307773113251, "SimDiff": 0.11247619241476059}, {"lex": "distancing", "Coronavirus": 0.050652213394641876, "conspiracy": -0.0580480694770813, "SimDiff": 0.10870028287172318}, {"lex": "quarantines", "Coronavirus": 0.0307206679135561, "conspiracy": -0.06277474761009216, "SimDiff": 0.09349541552364826}, {"lex": "research", "Coronavirus": 0.12641744315624237, "conspiracy": 0.03304028511047363, "SimDiff": 0.09337715804576874}, {"lex": "lockdowns", "Coronavirus": 0.023748569190502167, "conspiracy": -0.06043205037713051, "SimDiff": 0.08418061956763268}, {"lex": "rona", "Coronavirus": -0.014205201528966427, "conspiracy": -0.08521801233291626, "SimDiff": 0.07101281080394983}, {"lex": "science", "Coronavirus": -0.008252025581896305, "conspiracy": -0.047907549887895584, "SimDiff": 0.03965552430599928}, {"lex": "pandemic", "Coronavirus": -0.02237873524427414, "conspiracy": -0.05953046306967735, "SimDiff": 0.037151727825403214}, {"lex": "moderna", "Coronavirus": 0.08282074332237244, "conspiracy": 0.050235699862241745, "SimDiff": 0.03258504346013069}, {"lex": "maskless", "Coronavirus": -0.031929414719343185, "conspiracy": -0.06295277178287506, "SimDiff": 0.031023357063531876}, {"lex": "spreader", "Coronavirus": -0.14250540733337402, "conspiracy": -0.12802410125732422, "SimDiff": -0.014481306076049805}, {"lex": "sars", "Coronavirus": -0.07047674059867859, "conspiracy": -0.05521240830421448, "SimDiff": -0.015264332294464111}, {"lex": "corona", "Coronavirus": -0.16207775473594666, "conspiracy": -0.1439971923828125, "SimDiff": -0.018080562353134155}]}}, {"mode": "vega-lite"});
</script>



```python
# proj_sims_chart.save(f'../out/proj-emb_{models[0]["name"]}--{models[1]["name"]}___{pole_words[0]}--{pole_words[1]}.pdf')
```

#### _objective_ vs _subjective_

```python
pole_words = ['objective', 'subjective']
```

```python
proj_sims = get_axis_sims(lexs, models, pole_words, k=10)
```

```python
proj_sims = aggregate_proj_sims(proj_sims)
```

```python
proj_sims_chart = plot_sem_axis(proj_sims, models)
proj_sims_chart
```





<div id="altair-viz-c5c1148f466943cbb795da46a70f651a"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-c5c1148f466943cbb795da46a70f651a") {
      outputDiv = document.getElementById("altair-viz-c5c1148f466943cbb795da46a70f651a");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": {"type": "line", "color": "red", "point": {"color": "red"}}, "encoding": {"x": {"field": "conspiracy", "title": "r/conspiracy: red", "type": "quantitative"}, "y": {"field": "lex", "sort": {"field": "SimDiff"}, "title": "", "type": "nominal"}}}, {"mark": {"type": "line", "color": "blue", "point": {"color": "blue"}}, "encoding": {"x": {"field": "Coronavirus", "title": "r/Coronavirus: blue", "type": "quantitative"}, "y": {"field": "lex", "sort": {"field": "SimDiff"}, "title": "", "type": "nominal"}}}], "data": {"name": "data-2911885988033a2c102b5d06089afd11"}, "title": "", "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-2911885988033a2c102b5d06089afd11": [{"lex": "science", "Coronavirus": 0.35689404606819153, "conspiracy": 0.014754777774214745, "SimDiff": 0.3421392682939768}, {"lex": "research", "Coronavirus": 0.23729760944843292, "conspiracy": 0.07461787760257721, "SimDiff": 0.1626797318458557}, {"lex": "vaccines", "Coronavirus": 0.08601615577936172, "conspiracy": -0.046296343207359314, "SimDiff": 0.13231249898672104}, {"lex": "sars", "Coronavirus": 0.07451134920120239, "conspiracy": -0.021860774606466293, "SimDiff": 0.09637212380766869}, {"lex": "distancing", "Coronavirus": 0.032162513583898544, "conspiracy": -0.062317121773958206, "SimDiff": 0.09447963535785675}, {"lex": "corona", "Coronavirus": 0.015173436142504215, "conspiracy": -0.07627764344215393, "SimDiff": 0.09145107958465815}, {"lex": "pandemic", "Coronavirus": 0.032166216522455215, "conspiracy": -0.049691323190927505, "SimDiff": 0.08185753971338272}, {"lex": "lockdowns", "Coronavirus": 0.05210113525390625, "conspiracy": -0.016655955463647842, "SimDiff": 0.06875709071755409}, {"lex": "rona", "Coronavirus": 0.01636863872408867, "conspiracy": -0.04496964067220688, "SimDiff": 0.06133827939629555}, {"lex": "masks", "Coronavirus": 0.04734383150935173, "conspiracy": -0.0035551167093217373, "SimDiff": 0.05089894821867347}, {"lex": "spreader", "Coronavirus": -0.05658354610204697, "conspiracy": -0.08908845484256744, "SimDiff": 0.03250490874052048}, {"lex": "maskless", "Coronavirus": 0.0035706909839063883, "conspiracy": -0.017772605642676353, "SimDiff": 0.021343296626582742}, {"lex": "quarantines", "Coronavirus": 0.012220581993460655, "conspiracy": 0.01386572141200304, "SimDiff": -0.001645139418542385}, {"lex": "moderna", "Coronavirus": 0.022082751616835594, "conspiracy": 0.029249070212244987, "SimDiff": -0.007166318595409393}]}}, {"mode": "vega-lite"});
</script>



```python
# proj_sims_chart.save(f'../out/proj-emb_{models[0]["name"]}--{models[1]["name"]}___{pole_words[0]}--{pole_words[1]}.pdf')
```
