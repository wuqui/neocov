# NeoCov
> Semantic change and social semantic variation of Covid-related English neologisms on Reddit.


```python
%load_ext autoreload
%autoreload 2
```

```python
# all_data
```

```python
from neocov.read_data import *
from neocov.preproc import *
from neocov.type_emb import *
from neocov.communities import *
```

```python
from gensim.models import Word2Vec
import pandas as pd
from pathlib import Path
```

```python
DATA_DIR = '../data/'
COMMENTS_DIAC_DIR = f'{DATA_DIR}comments/by_date/'
OUT_DIR = '../out/'
```

## Semantic change

```python
YEAR = '2020'
```

### Read data

#### Get file paths

```python
comment_paths_year = get_comments_paths_year(COMMENTS_DIAC_DIR, YEAR)
```

#### Read comments

```python
%%time
comments = read_comm_csvs(comment_paths_year)
```

```python
comments
```

### Pre-process comments

```python
%%time
comments_clean = clean_comments(comments)
```

```python
docs = comments_clean['body'].to_list()
```

```python
import pickle
```

```python
with open(f'{OUT_DIR}docs_clean/diac_{YEAR}.pickle', 'wb') as fp:
    pickle.dump(docs, fp)
```

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
model_2019 = Word2Vec.load(f'{OUT_DIR}models/2019_ep-20.model')
```

```python
model_2020 = Word2Vec.load(f'{OUT_DIR}models/2020_ep-20.model')
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





    <gensim.models.word2vec.Word2Vec at 0x110df3160>



```python
assert len(model_2019.wv.key_to_index) == len(model_2020.wv.vectors)
```

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

```python
distances = measure_distances(model_2019, model_2020)
```

TODO: filter by true type frequency; `Gensim`'s type frequency seems incorrect; it probably reflects frequency ranks instead of total counts.

```python
blacklist_lex = (pd.read_csv('../data/blacklist_lex.csv')
    .query('Excl == True')
    .loc[:, 'Lex']
)
```

```python
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
      <td>991</td>
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
      <td>120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>childe</td>
      <td>0.980564</td>
      <td>209</td>
      <td>222</td>
    </tr>
    <tr>
      <th>4</th>
      <td>megalodon</td>
      <td>0.975273</td>
      <td>752</td>
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
      <td>148</td>
      <td>159</td>
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
)
```

```python
sem_change_cands_out.to_csv(
        f'{OUT_DIR}sem_change_cands.csv',
        columns=['', 'Lexeme', 'SemDist'],
        index=False
    )
```

### Inspect nearest neighbours of lexemes

```python
LEX_NBS = 'ahahahah'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=1,
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
      <th>model</th>
      <th>lex</th>
      <th>similarity</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>hahahha</td>
      <td>0.455687</td>
      <td>76</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>ahaha</td>
      <td>0.455320</td>
      <td>668</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>hahaha</td>
      <td>0.441289</td>
      <td>6690</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>ahha</td>
      <td>0.436381</td>
      <td>43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>myyyy</td>
      <td>0.434361</td>
      <td>14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>funni</td>
      <td>0.433964</td>
      <td>49</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>hahahah</td>
      <td>0.432263</td>
      <td>496</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>yeaaaa</td>
      <td>0.429689</td>
      <td>33</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>yess</td>
      <td>0.429398</td>
      <td>320</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>woooow</td>
      <td>0.427130</td>
      <td>46</td>
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
      <th>model</th>
      <th>lex</th>
      <th>similarity</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>100000</th>
      <td>2</td>
      <td>wiki_rule_2</td>
      <td>0.389549</td>
      <td>127</td>
    </tr>
    <tr>
      <th>100001</th>
      <td>2</td>
      <td>jk</td>
      <td>0.381757</td>
      <td>2248</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>dqw4w9wgxcq</td>
      <td>0.348253</td>
      <td>763</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>2</td>
      <td>wiki_rule_b</td>
      <td>0.346046</td>
      <td>12</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>2</td>
      <td>20enabled</td>
      <td>0.326490</td>
      <td>476</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>20questions</td>
      <td>0.314265</td>
      <td>513</td>
    </tr>
    <tr>
      <th>100006</th>
      <td>2</td>
      <td>flowerboy</td>
      <td>0.309573</td>
      <td>11</td>
    </tr>
    <tr>
      <th>100007</th>
      <td>2</td>
      <td>_love_</td>
      <td>0.307687</td>
      <td>21</td>
    </tr>
    <tr>
      <th>100008</th>
      <td>2</td>
      <td>subed</td>
      <td>0.306114</td>
      <td>21</td>
    </tr>
    <tr>
      <th>100009</th>
      <td>2</td>
      <td>420th</td>
      <td>0.303107</td>
      <td>22</td>
    </tr>
  </tbody>
</table>
</div>


Not related to Covid:

- sunsetting: > gaming-related meaning in 2020
- childe: > gaming-related proper name in 2020
- megalodon: > gaming-related proper name in 2020
- newf: (derogatory) slang term for people from Newfoundland (Canada)
- chaz: > Capitol Hill Autonomous Zone (CHAZ)
- klee: > computer game character, proper name
- rittenhouse: whiskey brand > proper name, involved in shooting related to BLM protests

Related to Covid:

- cerb: > Canada Emergency Response Benefit for Covid
- vacuo: > medical term, 'vacuum'
- moderna: > vaccine

## Social semantic variation

### Inspect subreddits

#### read comments

```python
comments_dir_path = Path('../data/comments/lexeme/')
```

```python
comments_paths = list(comments_dir_path.glob(f'Covid*.csv'))
```

```python
%%time
comments = read_comm_csvs(comments_paths)
comments
```

TODO: filter comments

- [ ] remove duplicates
- [ ] remove bots

#### get subreddit counts

```python
subr_counts = get_subr_counts(comments)
```

```python
subr_counts_plt = plot_subr_counts(subr_counts, k=20)
subr_counts_plt
```

```python
subr_counts_plt.save(f'{OUT_DIR}subr_counts.png', scale_factor=2.0)
```

### Train models

```python
COMMENTS_DIR_SUBR = '../data/comments/subr/'
```

```python
SUBR = 'conspiracy'
```

```python
fpaths = get_comments_paths_subr(COMMENTS_DIR_SUBR, SUBR)
```

```python
%%time
comments = read_comm_csvs(fpaths)
```

```python
%%time
comments_clean = clean_comments(comments)
```

```python
docs = comments_clean['body']
```

```python
docs = docs.to_list()
```

```python
import pickle
```

```python
with open(f'{OUT_DIR}docs_clean/subr_{SUBR}.pickle', 'wb') as fp:
    pickle.dump(docs, fp)
```

```python
with open('{OUT_DIR}docs_clean/subr_{SUBR}.pickle', 'rb') as fp:
    docs = pickle.load(fp)
```

Corpus information

| Subreddit          | Comments  | DateFirst  | DateLast   |
|:-------------------|---------: |:-----------|:-----------|
| LockdownSkepticism |   520,392 | 2020-03-26 | 2020-12-27 |  
| Coronavirus        | 4,121,144 | 2020-01-21 | 2020-12-27 |
| conspiracy         | 3,973,514 | 2020-01-01 | 2020-12-27 |

```python
corpus = Corpus(docs)
```

```python
%%time
model = train_model(corpus)
```

```python
len(model.wv.key_to_index)
```

```python
model.save(f'{OUT_DIR}models/{SUBR}.model')
```

### Load models

```python
SUBRS = ['Coronavirus', 'conspiracy']
```

```python
model_1 = Word2Vec.load(f'{OUT_DIR}models/{SUBRS[0]}.model')
```

```python
model_2 = Word2Vec.load(f'{OUT_DIR}models/{SUBRS[1]}.model')
```

### Align models

```python
model_1_vocab = len(model_1.wv.key_to_index)
model_2_vocab = len(model_2.wv.key_to_index)
```

```python
smart_procrustes_align_gensim(model_1, model_2)
```

```python
assert len(model_1.wv.key_to_index) == len(model_2.wv.vectors)
```

```python
models_vocab = pd.DataFrame(
    columns=['Model', 'Words'],
    data=[
        [SUBRS[0], model_1_vocab],
        [SUBRS[1], model_2_vocab],
        ['intersection', len(model_1.wv.key_to_index)]
    ],
)

models_vocab
```

```python
models_vocab.to_csv(f'{OUT_DIR}models_subrs_vocab.csv', index=False)
```

### Measure distances

```python
distances = measure_distances(model_1, model_2)
```

#### words that differ the most between both communities

```python
freq_min = 100

distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')\
    .sort_values('dist_sem', ascending=False)\
    .head(20)
```

#### nearest neighbours for target lexemes in both communities

```python
LEX = 'vaccine'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX, 
    freq_min=100,
    model_1=model_1, 
    model_2=model_2,
    k=10
)

display(nbs_model_1, nbs_model_2)
```

#### biggest discrepancies in nearest neighbours for target lexemes

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX, 
    freq_min=150,
    model_1=model_1, 
    model_2=model_2,
    k=100_000
)
```

```python
nbs_diffs = pd.merge(
    nbs_model_1, nbs_model_2, 
    on='lex',
    suffixes = ('_1', '_2')
)
```

```python
nbs_diffs = nbs_diffs\
    .assign(sim_diff = abs(nbs_diffs['similarity_1'] - nbs_diffs['similarity_2']))\
    .sort_values('sim_diff', ascending=False)\
    .reset_index(drop=True)\
    .query('lex.str.len() >= 4')
```

```python
topn = 10

subr_1_nbs = nbs_diffs\
    .query('similarity_1 > similarity_2')\
    .nlargest(topn, 'sim_diff')

subr_2_nbs = nbs_diffs\
    .query('similarity_2 > similarity_1')\
    .nlargest(topn, 'sim_diff')

display(subr_1_nbs, subr_2_nbs)
```
