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
pd.set_option('display.max_rows', 100)
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

```python
models_vocab.to_csv(f'{OUT_DIR}models_vocab.csv', index=False)
```

### Measure distances

```python
distances = measure_distances(model_2019, model_2020)
```

TODO: filter by true type frequency; `Gensim`'s type frequency seems incorrect; it probably reflects frequency ranks instead of total counts.

```python
blacklist_lex = load_blacklist_lex()

k = 500
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

    67181 67181
    67181 67181





    <gensim.models.word2vec.Word2Vec at 0x187e17c40>



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
    <tr>
      <th>2</th>
      <td>intersection</td>
      <td>67181</td>
    </tr>
  </tbody>
</table>
</div>



```python
models_vocab.to_csv(f'{OUT_DIR}models_subrs_vocab.csv', index=False)
```

### Measure distances

```python
distances = measure_distances(model_1, model_2)
```

#### words that differ the most between both communities

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
      <td>0.989504</td>
      <td>2233</td>
      <td>2474</td>
    </tr>
    <tr>
      <th>1</th>
      <td>resubmit</td>
      <td>0.944866</td>
      <td>1763</td>
      <td>1928</td>
    </tr>
    <tr>
      <th>2</th>
      <td>waiters</td>
      <td>0.929376</td>
      <td>149</td>
      <td>206</td>
    </tr>
    <tr>
      <th>3</th>
      <td>curiously</td>
      <td>0.928779</td>
      <td>118</td>
      <td>164</td>
    </tr>
    <tr>
      <th>4</th>
      <td>subsequently</td>
      <td>0.924492</td>
      <td>12174</td>
      <td>11956</td>
    </tr>
    <tr>
      <th>5</th>
      <td>anons</td>
      <td>0.920569</td>
      <td>260</td>
      <td>351</td>
    </tr>
    <tr>
      <th>6</th>
      <td>blacklivesmatter</td>
      <td>0.918460</td>
      <td>107</td>
      <td>150</td>
    </tr>
    <tr>
      <th>7</th>
      <td>redpill</td>
      <td>0.914164</td>
      <td>119</td>
      <td>165</td>
    </tr>
    <tr>
      <th>8</th>
      <td>derek</td>
      <td>0.913568</td>
      <td>211</td>
      <td>287</td>
    </tr>
    <tr>
      <th>9</th>
      <td>borderline</td>
      <td>0.892894</td>
      <td>21891</td>
      <td>20974</td>
    </tr>
    <tr>
      <th>10</th>
      <td>submissions</td>
      <td>0.890047</td>
      <td>3777</td>
      <td>4093</td>
    </tr>
    <tr>
      <th>11</th>
      <td>promotional</td>
      <td>0.887624</td>
      <td>344</td>
      <td>450</td>
    </tr>
    <tr>
      <th>12</th>
      <td>mena</td>
      <td>0.886735</td>
      <td>196</td>
      <td>267</td>
    </tr>
    <tr>
      <th>13</th>
      <td>nicaraguan</td>
      <td>0.885770</td>
      <td>111</td>
      <td>154</td>
    </tr>
    <tr>
      <th>14</th>
      <td>acorn</td>
      <td>0.885177</td>
      <td>105</td>
      <td>147</td>
    </tr>
    <tr>
      <th>15</th>
      <td>greer</td>
      <td>0.883525</td>
      <td>105</td>
      <td>147</td>
    </tr>
    <tr>
      <th>16</th>
      <td>goyim</td>
      <td>0.879430</td>
      <td>287</td>
      <td>386</td>
    </tr>
    <tr>
      <th>17</th>
      <td>snowfall</td>
      <td>0.873538</td>
      <td>208</td>
      <td>283</td>
    </tr>
    <tr>
      <th>18</th>
      <td>goodyear</td>
      <td>0.869714</td>
      <td>101</td>
      <td>141</td>
    </tr>
    <tr>
      <th>19</th>
      <td>submitter</td>
      <td>0.869147</td>
      <td>260</td>
      <td>352</td>
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
sem_change_cands_out.to_csv(
        f'{OUT_DIR}sem_var_soc_cands.csv',
        columns=['', 'Lexeme', 'SemDist'],
        index=False
    )
```

#### nearest neighbours for target lexemes in both communities

```python
LEX_NBS = 'lockdown'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=50,
    model_1=model_1, 
    model_2=model_2,
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
      <td>shutdown</td>
      <td>0.844873</td>
      <td>5413</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>lockdowns</td>
      <td>0.768522</td>
      <td>16637</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>shutdowns</td>
      <td>0.661069</td>
      <td>1632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>curfew</td>
      <td>0.613175</td>
      <td>1079</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>quarantine</td>
      <td>0.600933</td>
      <td>26502</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>restrictions</td>
      <td>0.581443</td>
      <td>16482</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>quarantines</td>
      <td>0.568791</td>
      <td>1634</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>quarentine</td>
      <td>0.540457</td>
      <td>255</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>curfews</td>
      <td>0.536410</td>
      <td>456</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>containment</td>
      <td>0.515410</td>
      <td>2526</td>
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
      <th>29861</th>
      <td>2</td>
      <td>lockdowns</td>
      <td>0.794638</td>
      <td>15881</td>
    </tr>
    <tr>
      <th>29862</th>
      <td>2</td>
      <td>shutdown</td>
      <td>0.726183</td>
      <td>5529</td>
    </tr>
    <tr>
      <th>29863</th>
      <td>2</td>
      <td>quarantine</td>
      <td>0.723483</td>
      <td>24948</td>
    </tr>
    <tr>
      <th>29864</th>
      <td>2</td>
      <td>shutdowns</td>
      <td>0.702284</td>
      <td>1805</td>
    </tr>
    <tr>
      <th>29865</th>
      <td>2</td>
      <td>quarantines</td>
      <td>0.622162</td>
      <td>1806</td>
    </tr>
    <tr>
      <th>29866</th>
      <td>2</td>
      <td>curfew</td>
      <td>0.595864</td>
      <td>1236</td>
    </tr>
    <tr>
      <th>29867</th>
      <td>2</td>
      <td>restrictions</td>
      <td>0.583930</td>
      <td>15817</td>
    </tr>
    <tr>
      <th>29868</th>
      <td>2</td>
      <td>curfews</td>
      <td>0.581970</td>
      <td>577</td>
    </tr>
    <tr>
      <th>29869</th>
      <td>2</td>
      <td>pandemic</td>
      <td>0.580793</td>
      <td>67364</td>
    </tr>
    <tr>
      <th>29870</th>
      <td>2</td>
      <td>quarentine</td>
      <td>0.528733</td>
      <td>346</td>
    </tr>
  </tbody>
</table>
</div>


#### embeddings projection

```python
from scipy import spatial
import altair as alt
import numpy as np
```

```python
models = []
models.append({'subreddit': SUBRS[0], 'model': model_1})
models.append({'subreddit': SUBRS[1], 'model': model_2})
```

```python
def get_pole_avg(model, lex, k):
	vecs = []
	vecs.append(model.wv[lex])
	for closest_word, similarity in model.wv.most_similar(positive=lex, topn=k):
		vecs.append(model.wv[closest_word])
	pole_avg = np.mean(vecs, axis=0)
	return pole_avg
```

```python
def make_sem_axis(model, pole_word_1: str, pole_word_2: str):
	pole_1_vec = model_1.wv.get_vector(pole_1)
	pole_2_vec = model_1.wv.get_vector(pole_2)
	sem_axis = pole_1_vec - pole_2_vec
	return sem_axis
```

```python
def make_sem_axis_avg(model, pole_word_1: str, pole_word_2: str):
	pole_1_vec = model_1.wv.get_vector(pole_1)
	pole_2_vec = model_1.wv.get_vector(pole_2)
	pole_1_avg = get_pole_avg(model_1, pole_word_1, k=10)
	pole_2_avg = get_pole_avg(model_1, pole_word_2, k=10)
	sem_axis = pole_1_avg - pole_2_avg
	return sem_axis
```

```python
def get_axis_sim(lex: str, pole_word_1: str, pole_word_2: str, model):
	sem_axis = make_sem_axis_avg(model, pole_word_1, pole_word_2)
	lex_vec = model.wv.get_vector(lex)
	sim_cos = 1 - spatial.distance.cosine(lex_vec, sem_axis)
	return sim_cos
```

```python
lex = 'lockdown'
pole_1 = 'good'
pole_2 = 'bad'
```

```python
lexs = [
	'lockdown', 'lockdowns', 
	'shutdown', 'shutdowns', 
	'vaccine', 'vaccines', 
	'mask', 'masks',
	'order', 'police', 'science', 'mandate',
	'thing', 'tree', 
	'give', 'take'
	# 'happy', 'sad',
	]
```

```python
sims = []
for lex in lexs:
	for model in models:
		sim = {}
		sim['subreddit'] = model['subreddit']
		sim['lex'] = lex
		sim['sim'] = get_axis_sim(lex, pole_1, pole_2, model['model'])
		sims.append(sim)
```

```python
sims_df = pd.DataFrame(sims)

alt.Chart(sims_df).mark_line(point=True).encode(
	x='sim',
	y=alt.Y('lex', sort=None),
	color='subreddit'
)

```





<div id="altair-viz-b3abe0aaafb14650ac220c6fa898517c"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-b3abe0aaafb14650ac220c6fa898517c") {
      outputDiv = document.getElementById("altair-viz-b3abe0aaafb14650ac220c6fa898517c");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-aab3a02273668f9c5dc01b4c2a123765"}, "mark": {"type": "line", "point": true}, "encoding": {"color": {"field": "subreddit", "type": "nominal"}, "x": {"field": "sim", "type": "quantitative"}, "y": {"field": "lex", "sort": null, "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-aab3a02273668f9c5dc01b4c2a123765": [{"subreddit": "Coronavirus", "lex": "lockdown", "sim": -0.020428422838449478}, {"subreddit": "conspiracy", "lex": "lockdown", "sim": -0.11954280734062195}, {"subreddit": "Coronavirus", "lex": "lockdowns", "sim": -0.03195760399103165}, {"subreddit": "conspiracy", "lex": "lockdowns", "sim": -0.09696231037378311}, {"subreddit": "Coronavirus", "lex": "shutdown", "sim": -0.0030814199708402157}, {"subreddit": "conspiracy", "lex": "shutdown", "sim": -0.07952913641929626}, {"subreddit": "Coronavirus", "lex": "shutdowns", "sim": -0.061103351414203644}, {"subreddit": "conspiracy", "lex": "shutdowns", "sim": -0.09631608426570892}, {"subreddit": "Coronavirus", "lex": "vaccine", "sim": 0.027036398649215698}, {"subreddit": "conspiracy", "lex": "vaccine", "sim": -0.057832762598991394}, {"subreddit": "Coronavirus", "lex": "vaccines", "sim": 0.02113475278019905}, {"subreddit": "conspiracy", "lex": "vaccines", "sim": -0.025540443137288094}, {"subreddit": "Coronavirus", "lex": "mask", "sim": 0.02409209869801998}, {"subreddit": "conspiracy", "lex": "mask", "sim": -0.06673668324947357}, {"subreddit": "Coronavirus", "lex": "masks", "sim": 0.042272794991731644}, {"subreddit": "conspiracy", "lex": "masks", "sim": -0.022345764562487602}, {"subreddit": "Coronavirus", "lex": "order", "sim": 0.03343459591269493}, {"subreddit": "conspiracy", "lex": "order", "sim": -0.013651296496391296}, {"subreddit": "Coronavirus", "lex": "police", "sim": 0.05815239995718002}, {"subreddit": "conspiracy", "lex": "police", "sim": 0.009731519035995007}, {"subreddit": "Coronavirus", "lex": "science", "sim": -0.04062918573617935}, {"subreddit": "conspiracy", "lex": "science", "sim": -0.01712346449494362}, {"subreddit": "Coronavirus", "lex": "mandate", "sim": 0.050309017300605774}, {"subreddit": "conspiracy", "lex": "mandate", "sim": 0.023100711405277252}, {"subreddit": "Coronavirus", "lex": "thing", "sim": -0.09180065989494324}, {"subreddit": "conspiracy", "lex": "thing", "sim": -0.06663163006305695}, {"subreddit": "Coronavirus", "lex": "tree", "sim": -0.00044312604586593807}, {"subreddit": "conspiracy", "lex": "tree", "sim": 0.0230832789093256}, {"subreddit": "Coronavirus", "lex": "give", "sim": 0.04639571160078049}, {"subreddit": "conspiracy", "lex": "give", "sim": 0.029668521136045456}, {"subreddit": "Coronavirus", "lex": "take", "sim": -0.0009267961140722036}, {"subreddit": "conspiracy", "lex": "take", "sim": -0.025768481194972992}]}}, {"mode": "vega-lite"});
</script>



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
