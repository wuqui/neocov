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
```

```python
import pandas as pd
```

## Variables

```python
DATA_DIR = '../data/'
COMMENTS_DIAC_DIR = f'{DATA_DIR}comments/by_date/'
OUT_DIR = '..out/'
```

## Read data

### Get file paths

```python
YEAR = '2020'
```

```python
comment_paths_year = get_comments_paths_year(COMMENTS_DIAC_DIR, YEAR)
```

### Read comments

```python
%%time
comments = read_comm_csvs(comment_paths_year)
```

```python
comments
```

```python
comments.value_counts('subreddit')
```

## Pre-process comments

### run preprocessing

```python
%%time
comments = clean_comments(comments)
```

## Train models

### Create corpus

```python
class Corpus:
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, docs_clean):
        self.docs_clean = docs_clean

    def __iter__(self):
        for doc in self.docs_clean:
            yield doc
```

```python
corpus = Corpus(comments['body'])
```

### Train model

```python
from gensim.models import Word2Vec
```

```python
def train_emb(corpus, 
              MIN_COUNT=5, 
              SIZE=300, 
              WORKERS=8, 
              WINDOW=5):
    model = Word2Vec(
        corpus, 
        min_count=MIN_COUNT,
        vector_size=SIZE,
        workers=WORKERS, 
        window=WINDOW
    )
    return model
```

```python
%%time
model = train_emb(corpus)
```

```python
len(model.wv.key_to_index)
```

### Save model

```python
model.save(f'{OUT_DIR}models/{YEAR}.model')
```

### Load models

```python
model_2019 = Word2Vec.load('{OUT_DIR}models/2019.model')
```

```python
model_2020 = Word2Vec.load('{OUT_DIR}models/2020.model')
```

## Align models

```python
import numpy as np
```

```python
def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.wv.index_to_key)
    vocab_m2 = set(m2.wv.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.wv.get_vecattr(w, "count") + m2.wv.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.wv.key_to_index[w] for w in common_vocab]
        old_arr = m.wv.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.wv.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.wv.key_to_index = new_key_to_index
        m.wv.index_to_key = new_index_to_key
        
        print(len(m.wv.key_to_index), len(m.wv.vectors))
        
    return (m1,m2)
```

```python
def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)

    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.wv.get_normed_vectors()
    other_vecs = in_other_embed.wv.get_normed_vectors()

    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.wv.vectors = (other_embed.wv.vectors).dot(ortho)    
    
    return other_embed
```

```python
smart_procrustes_align_gensim(model_2019, model_2020)
```

    190756 190756
    190756 190756





    <gensim.models.word2vec.Word2Vec at 0x141570640>



```python
import pandas as pd
```

```python
models_vocab = pd.DataFrame(
    data=[
        ['2019', len(model_2019.wv.key_to_index)],
        ['2020', len(model_2020.wv.key_to_index)],
        ['intersection', len(set(model_2019.wv.key_to_index).intersection(set(model_2020.wv.key_to_index)))]
    ],
    columns=['', 'words']
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
      <th></th>
      <th>words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2019</td>
      <td>190756</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020</td>
      <td>190756</td>
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

## Measure distances

```python
from scipy import spatial
```

```python
def measure_distances(model_1, model_2):
    distances = pd.DataFrame(
    data=(
            #[w, spatial.distance.euclidean(model_1.wv[w], model_2.wv[w]), 
            #[w, np.sum(model_1.wv[w] * model_2.wv[w]) / (np.linalg.norm(model_1.wv[w]) * np.linalg.norm(model_2.wv[w])), 
            [w, spatial.distance.cosine(model_1.wv[w], model_2.wv[w]), 
             model_1.wv.get_vecattr(w, "count"), 
             model_2.wv.get_vecattr(w, "count")
            ] for w in model_1.wv.index_to_key
        ), 
        columns = ('lex', 'dist_sem', "freq_1", "freq_2")
    )
    return distances
```

```python
distances = measure_distances(model_2019, model_2020)
```

```python
distances\
    .sort_values('dist_sem', ascending=False)

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
      <th>181299</th>
      <td>financiados</td>
      <td>1.270406</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>165232</th>
      <td>______________________________________________...</td>
      <td>1.257892</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>181454</th>
      <td>2ffireemblem</td>
      <td>1.247719</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>189647</th>
      <td>obedece</td>
      <td>1.239514</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>126402</th>
      <td>1281</td>
      <td>1.218590</td>
      <td>14</td>
      <td>16</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>175</th>
      <td>years</td>
      <td>0.027202</td>
      <td>175105</td>
      <td>192696</td>
    </tr>
    <tr>
      <th>171086</th>
      <td>ppx_yo_dt_b_asin_title_o09_s00</td>
      <td>0.025620</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>46607</th>
      <td>imagestabilization</td>
      <td>0.025614</td>
      <td>85</td>
      <td>92</td>
    </tr>
    <tr>
      <th>144119</th>
      <td>ppx_yo_dt_b_asin_title_o03_s00</td>
      <td>0.018814</td>
      <td>11</td>
      <td>13</td>
    </tr>
    <tr>
      <th>68529</th>
      <td>u5e9htr</td>
      <td>0.012333</td>
      <td>42</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
<p>190756 rows × 4 columns</p>
</div>



```python
def get_sem_change_cands(distances, k=20, freq_min=1):
    sem_change_cands = (distances
        .query('freq_1 > @freq_min and freq_2 > @freq_min')
        .query('lex.str.isalpha() == True')
        .query('lex.str.len() > 3')
        .nlargest(k, 'dist_sem')
        .reset_index(drop=True)
        )
    return sem_change_cands
```

```python
sem_change_cands = get_sem_change_cands(distances, k=100, freq_min=1000)
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
      <td>corona</td>
      <td>0.927504</td>
      <td>3553</td>
      <td>3684</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pandemic</td>
      <td>0.912615</td>
      <td>9504</td>
      <td>9957</td>
    </tr>
    <tr>
      <th>2</th>
      <td>snapchatting</td>
      <td>0.912304</td>
      <td>2262</td>
      <td>2345</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dodo</td>
      <td>0.864197</td>
      <td>1651</td>
      <td>1716</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rubric</td>
      <td>0.839424</td>
      <td>1058</td>
      <td>1109</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>neon</td>
      <td>0.393886</td>
      <td>1326</td>
      <td>1391</td>
    </tr>
    <tr>
      <th>96</th>
      <td>villagers</td>
      <td>0.393821</td>
      <td>1274</td>
      <td>1333</td>
    </tr>
    <tr>
      <th>97</th>
      <td>goose</td>
      <td>0.391982</td>
      <td>1197</td>
      <td>1260</td>
    </tr>
    <tr>
      <th>98</th>
      <td>mute</td>
      <td>0.391320</td>
      <td>5323</td>
      <td>5505</td>
    </tr>
    <tr>
      <th>99</th>
      <td>lastly</td>
      <td>0.389894</td>
      <td>2129</td>
      <td>2193</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 4 columns</p>
</div>



```python
sem_change_cands_out = sem_change_cands\
    .nlargest(100, 'dist_sem')\
    .assign(index_1 = lambda df: df.index + 1)\
    .assign(dist_sem = lambda df: df['dist_sem'].round(2))\
    .assign(dist_sem = lambda df: df['dist_sem'].apply('{:.2f}'.format))\
    .rename({'index_1': '', 'lex': 'Lexeme', 'dist_sem': 'SemDist'}, axis=1)

sem_change_cands_out.head(20)
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
      <th>freq_1</th>
      <th>freq_2</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>corona</td>
      <td>0.93</td>
      <td>3553</td>
      <td>3684</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pandemic</td>
      <td>0.91</td>
      <td>9504</td>
      <td>9957</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>snapchatting</td>
      <td>0.91</td>
      <td>2262</td>
      <td>2345</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dodo</td>
      <td>0.86</td>
      <td>1651</td>
      <td>1716</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rubric</td>
      <td>0.84</td>
      <td>1058</td>
      <td>1109</td>
      <td>5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nices</td>
      <td>0.81</td>
      <td>7457</td>
      <td>7710</td>
      <td>6</td>
    </tr>
    <tr>
      <th>6</th>
      <td>hyphens</td>
      <td>0.81</td>
      <td>1044</td>
      <td>1096</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>asterisks</td>
      <td>0.81</td>
      <td>1085</td>
      <td>1138</td>
      <td>8</td>
    </tr>
    <tr>
      <th>8</th>
      <td>distancing</td>
      <td>0.79</td>
      <td>2910</td>
      <td>3038</td>
      <td>9</td>
    </tr>
    <tr>
      <th>9</th>
      <td>newbies</td>
      <td>0.78</td>
      <td>1566</td>
      <td>1644</td>
      <td>10</td>
    </tr>
    <tr>
      <th>10</th>
      <td>delet</td>
      <td>0.78</td>
      <td>1269</td>
      <td>1329</td>
      <td>11</td>
    </tr>
    <tr>
      <th>11</th>
      <td>blah</td>
      <td>0.75</td>
      <td>3683</td>
      <td>3826</td>
      <td>12</td>
    </tr>
    <tr>
      <th>12</th>
      <td>tracing</td>
      <td>0.75</td>
      <td>1228</td>
      <td>1293</td>
      <td>13</td>
    </tr>
    <tr>
      <th>13</th>
      <td>warzone</td>
      <td>0.71</td>
      <td>1022</td>
      <td>1070</td>
      <td>14</td>
    </tr>
    <tr>
      <th>14</th>
      <td>yandex</td>
      <td>0.70</td>
      <td>1111</td>
      <td>1169</td>
      <td>15</td>
    </tr>
    <tr>
      <th>15</th>
      <td>gaysnapchat</td>
      <td>0.70</td>
      <td>5145</td>
      <td>5328</td>
      <td>16</td>
    </tr>
    <tr>
      <th>16</th>
      <td>lockdown</td>
      <td>0.70</td>
      <td>4494</td>
      <td>4642</td>
      <td>17</td>
    </tr>
    <tr>
      <th>17</th>
      <td>muda</td>
      <td>0.69</td>
      <td>3300</td>
      <td>3436</td>
      <td>18</td>
    </tr>
    <tr>
      <th>18</th>
      <td>specifying</td>
      <td>0.68</td>
      <td>1135</td>
      <td>1191</td>
      <td>19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>origi</td>
      <td>0.67</td>
      <td>1219</td>
      <td>1282</td>
      <td>20</td>
    </tr>
  </tbody>
</table>
</div>



```python
sem_change_cands_out.to_csv(
        '{OUT_DIR}sem_change_cands.csv',
        columns=['', 'Lexeme', 'SemDist'],
        index=False
    )
```

## Inspect nearest neighbours

```python
LEX_NBS = 'distancing'
```

```python
def get_nearest_neighbours_models(lex, freq_min, model_1, model_2, topn=100_000):
    nbs = []
    for count, model in enumerate([model_1, model_2]):
        for nb, dist in model.wv.most_similar(lex, topn=topn):
            if model.wv.get_vecattr(nb, 'count') > freq_min:
                d = {}
                d['model'] = count + 1
                d['lex'] = nb
                d['similarity'] = dist
                d['freq'] = model.wv.get_vecattr(nb, "count")
                nbs.append(d)
    nbs_df = pd.DataFrame(nbs)
    nbs_df = nbs_df\
        .query('freq > @freq_min')\
        .groupby('model', group_keys=False)\
        .apply(lambda group: group.nlargest(10, 'similarity'))
    nbs_model_1 = nbs_df.query('model == 1')
    nbs_model_2 = nbs_df.query('model == 2')
    return nbs_model_1, nbs_model_2
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=50,
    model_1=model_2019, 
    model_2=model_2020
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
      <td>distanced</td>
      <td>0.837815</td>
      <td>309</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>disassociate</td>
      <td>0.717895</td>
      <td>93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>detaching</td>
      <td>0.685801</td>
      <td>61</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>deluding</td>
      <td>0.667654</td>
      <td>104</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>bettering</td>
      <td>0.633784</td>
      <td>198</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>incriminate</td>
      <td>0.629239</td>
      <td>80</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>isolating</td>
      <td>0.629057</td>
      <td>685</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>distract</td>
      <td>0.617911</td>
      <td>1553</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>handicapping</td>
      <td>0.610600</td>
      <td>54</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>detach</td>
      <td>0.603991</td>
      <td>244</td>
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
      <th>36688</th>
      <td>2</td>
      <td>distanced</td>
      <td>0.553989</td>
      <td>326</td>
    </tr>
    <tr>
      <th>36689</th>
      <td>2</td>
      <td>isolation</td>
      <td>0.547227</td>
      <td>2037</td>
    </tr>
    <tr>
      <th>36690</th>
      <td>2</td>
      <td>gatherings</td>
      <td>0.519332</td>
      <td>921</td>
    </tr>
    <tr>
      <th>36691</th>
      <td>2</td>
      <td>distance</td>
      <td>0.511493</td>
      <td>11355</td>
    </tr>
    <tr>
      <th>36692</th>
      <td>2</td>
      <td>lockdowns</td>
      <td>0.499619</td>
      <td>991</td>
    </tr>
    <tr>
      <th>36693</th>
      <td>2</td>
      <td>quarantines</td>
      <td>0.487039</td>
      <td>159</td>
    </tr>
    <tr>
      <th>36694</th>
      <td>2</td>
      <td>lockdown</td>
      <td>0.483064</td>
      <td>4642</td>
    </tr>
    <tr>
      <th>36695</th>
      <td>2</td>
      <td>masks</td>
      <td>0.477628</td>
      <td>8997</td>
    </tr>
    <tr>
      <th>36696</th>
      <td>2</td>
      <td>precautions</td>
      <td>0.469785</td>
      <td>1237</td>
    </tr>
    <tr>
      <th>36697</th>
      <td>2</td>
      <td>quarantine</td>
      <td>0.468756</td>
      <td>5225</td>
    </tr>
  </tbody>
</table>
</div>


```python
nbs_model_1.to_csv(f'{OUT_DIR}neighbours/{LEX_NBS}_2019.csv')
nbs_model_2.to_csv(f'{OUT_DIR}neighbours/{LEX_NBS}_2020.csv')
```

# Inspect subreddits

## read comments

```python
YEAR = 2019
```

```python
comments_paths = get_comments_paths_year(COMMENTS_DIAC_DIR, YEAR)
```

```python
%%time
comments = read_comm_csvs(comments_paths)
comments
```

    CPU times: user 47.7 s, sys: 6.22 s, total: 54 s
    Wall time: 54.8 s





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
      <td>Avinse</td>
      <td>Username Checks Out</td>
      <td>2019-05-07 21:11:36</td>
      <td>emrv0h9</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KeepingDankMemesDank</td>
      <td>If this is a dank meme, **Upvote** this commen...</td>
      <td>2019-05-07 21:11:37</td>
      <td>emrv0jp</td>
      <td>dankmemes</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UhPhrasing</td>
      <td>Just threaten them that you'll call the corpor...</td>
      <td>2019-05-07 21:11:37</td>
      <td>emrv0jq</td>
      <td>golf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[deleted]</td>
      <td>[removed]</td>
      <td>2019-05-07 21:11:37</td>
      <td>emrv0jr</td>
      <td>Barca</td>
    </tr>
    <tr>
      <th>4</th>
      <td>EnergetikNA</td>
      <td>honestly, do you really wanna go through an en...</td>
      <td>2019-05-07 21:11:37</td>
      <td>emrv0js</td>
      <td>soccer</td>
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
      <th>9599974</th>
      <td>DogBeersHadOne</td>
      <td>Guy who made the crossbuck had one job. One go...</td>
      <td>2019-06-19 21:59:59</td>
      <td>erl9mvx</td>
      <td>trains</td>
    </tr>
    <tr>
      <th>9599975</th>
      <td>VenomousCoffee</td>
      <td>Page number? Picture of the page?</td>
      <td>2019-06-19 21:59:59</td>
      <td>erl9mvw</td>
      <td>marvelstudios</td>
    </tr>
    <tr>
      <th>9599976</th>
      <td>Homerundude698</td>
      <td>So sexy baby</td>
      <td>2019-06-19 21:59:59</td>
      <td>erl9mvv</td>
      <td>gonewild30plus</td>
    </tr>
    <tr>
      <th>9599977</th>
      <td>CircusRama</td>
      <td>Removed for Rule 8</td>
      <td>2019-06-19 21:59:59</td>
      <td>erl9mwa</td>
      <td>fivenightsatfreddys</td>
    </tr>
    <tr>
      <th>9599978</th>
      <td>BusShelter</td>
      <td>This is barely even half an inch by the looks ...</td>
      <td>2019-06-19 21:59:59</td>
      <td>erl9muw</td>
      <td>soccer</td>
    </tr>
  </tbody>
</table>
<p>9599979 rows × 5 columns</p>
</div>



TODO: filter comments

- [ ] remove duplicates
- [ ] remove bots

## get subreddit counts

```python
def get_subr_counts(comments):
    subr_counts = comments\
        .groupby('subreddit')\
        .agg(comments_num = ('subreddit', 'count'))\
        .sort_values('comments_num', ascending=False)
    return subr_counts
```

```python
subr_counts = get_subr_counts(comments)
```

```python
import altair as alt
```

```python
def plot_subr_counts(subr_counts, k=20):
    chart = subr_counts\
        .reset_index()\
        .iloc[:k]\
        .pipe(alt.Chart)\
            .mark_bar()\
            .encode(
                x=alt.X('comments_num:Q'),
                y=alt.Y('subreddit:N', sort='-x')
            )
    return chart
```

```python
subr_counts_plt = plot_subr_counts(subr_counts, k=20)
subr_counts_plt
```





<div id="altair-viz-1d981c61e4bd4d54a9b18eeeaf7cd7ec"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-1d981c61e4bd4d54a9b18eeeaf7cd7ec") {
      outputDiv = document.getElementById("altair-viz-1d981c61e4bd4d54a9b18eeeaf7cd7ec");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-151658c588b77c0f8397fe41b7b06449"}, "mark": "bar", "encoding": {"x": {"field": "comments_num", "type": "quantitative"}, "y": {"field": "subreddit", "sort": "-x", "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-151658c588b77c0f8397fe41b7b06449": [{"subreddit": "AskReddit", "comments_num": 429516}, {"subreddit": "politics", "comments_num": 146023}, {"subreddit": "memes", "comments_num": 99027}, {"subreddit": "teenagers", "comments_num": 89685}, {"subreddit": "dankmemes", "comments_num": 84107}, {"subreddit": "PatriotsvsTexansLivz", "comments_num": 80476}, {"subreddit": "soccer", "comments_num": 73348}, {"subreddit": "nba", "comments_num": 69130}, {"subreddit": "AmItheAsshole", "comments_num": 65271}, {"subreddit": "funny", "comments_num": 64256}, {"subreddit": "NFLWeek14Lives", "comments_num": 60055}, {"subreddit": "nfl", "comments_num": 58602}, {"subreddit": "unpopularopinion", "comments_num": 55227}, {"subreddit": "worldnews", "comments_num": 54755}, {"subreddit": "ufc245fightonline", "comments_num": 48229}, {"subreddit": "The_Donald", "comments_num": 46733}, {"subreddit": "gaming", "comments_num": 45341}, {"subreddit": "UFC245StreamToday", "comments_num": 45162}, {"subreddit": "CFB", "comments_num": 44439}, {"subreddit": "news", "comments_num": 43733}]}}, {"mode": "vega-lite"});
</script>



```python
subr_counts_fname = 'Covid'
```

```python
subr_counts_plt.save(f'out/subr_counts_plt_{subr_counts_fname}.svg', scale_factor=2.0)
```

```python
comments\
    .query('subreddit == "hdsportsfeedtv"')\
     .sample(10)
```
