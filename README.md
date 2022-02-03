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
SUBRS = ['Coronavirus', 'LockdownSkepticism']
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

    37317 37317
    37317 37317





    <gensim.models.word2vec.Word2Vec at 0x187dbcfa0>



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
      <td>LockdownSkepticism</td>
      <td>38926</td>
    </tr>
    <tr>
      <th>2</th>
      <td>intersection</td>
      <td>37317</td>
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
      <td>plandemic</td>
      <td>0.892523</td>
      <td>789</td>
      <td>138</td>
    </tr>
    <tr>
      <th>1</th>
      <td>scams</td>
      <td>0.889811</td>
      <td>964</td>
      <td>167</td>
    </tr>
    <tr>
      <th>2</th>
      <td>vigorous</td>
      <td>0.866856</td>
      <td>647</td>
      <td>114</td>
    </tr>
    <tr>
      <th>3</th>
      <td>likewise</td>
      <td>0.843846</td>
      <td>1444</td>
      <td>251</td>
    </tr>
    <tr>
      <th>4</th>
      <td>borderline</td>
      <td>0.829629</td>
      <td>34936</td>
      <td>6561</td>
    </tr>
    <tr>
      <th>5</th>
      <td>examining</td>
      <td>0.827804</td>
      <td>1337</td>
      <td>234</td>
    </tr>
    <tr>
      <th>6</th>
      <td>review</td>
      <td>0.824861</td>
      <td>20052</td>
      <td>3647</td>
    </tr>
    <tr>
      <th>7</th>
      <td>improved</td>
      <td>0.822236</td>
      <td>17517</td>
      <td>3032</td>
    </tr>
    <tr>
      <th>8</th>
      <td>examination</td>
      <td>0.813457</td>
      <td>1314</td>
      <td>229</td>
    </tr>
    <tr>
      <th>9</th>
      <td>blurred</td>
      <td>0.807373</td>
      <td>634</td>
      <td>112</td>
    </tr>
    <tr>
      <th>10</th>
      <td>approved</td>
      <td>0.805417</td>
      <td>39951</td>
      <td>7422</td>
    </tr>
    <tr>
      <th>11</th>
      <td>soliciting</td>
      <td>0.805046</td>
      <td>3903</td>
      <td>692</td>
    </tr>
    <tr>
      <th>12</th>
      <td>cargo</td>
      <td>0.803016</td>
      <td>616</td>
      <td>108</td>
    </tr>
    <tr>
      <th>13</th>
      <td>choke</td>
      <td>0.800872</td>
      <td>1171</td>
      <td>205</td>
    </tr>
    <tr>
      <th>14</th>
      <td>internally</td>
      <td>0.799448</td>
      <td>574</td>
      <td>101</td>
    </tr>
    <tr>
      <th>15</th>
      <td>coverage</td>
      <td>0.794310</td>
      <td>10972</td>
      <td>1934</td>
    </tr>
    <tr>
      <th>16</th>
      <td>screened</td>
      <td>0.791834</td>
      <td>15463</td>
      <td>2759</td>
    </tr>
    <tr>
      <th>17</th>
      <td>mega</td>
      <td>0.791515</td>
      <td>2167</td>
      <td>389</td>
    </tr>
    <tr>
      <th>18</th>
      <td>speculate</td>
      <td>0.789921</td>
      <td>13655</td>
      <td>2379</td>
    </tr>
    <tr>
      <th>19</th>
      <td>reliable</td>
      <td>0.787391</td>
      <td>69524</td>
      <td>13660</td>
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
      <td>8598</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>lockdowns</td>
      <td>0.768522</td>
      <td>50035</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>shutdowns</td>
      <td>0.661069</td>
      <td>3037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>curfew</td>
      <td>0.613175</td>
      <td>1652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>quarantine</td>
      <td>0.600933</td>
      <td>40419</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>restrictions</td>
      <td>0.581443</td>
      <td>31971</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>quarantines</td>
      <td>0.568791</td>
      <td>2767</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>quarentine</td>
      <td>0.540457</td>
      <td>454</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>curfews</td>
      <td>0.536410</td>
      <td>751</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>containment</td>
      <td>0.515410</td>
      <td>4363</td>
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
      <td>0.737042</td>
      <td>9460</td>
    </tr>
    <tr>
      <th>29862</th>
      <td>2</td>
      <td>shutdown</td>
      <td>0.727802</td>
      <td>1489</td>
    </tr>
    <tr>
      <th>29863</th>
      <td>2</td>
      <td>lockdowners</td>
      <td>0.599373</td>
      <td>156</td>
    </tr>
    <tr>
      <th>29864</th>
      <td>2</td>
      <td>shutdowns</td>
      <td>0.584469</td>
      <td>545</td>
    </tr>
    <tr>
      <th>29865</th>
      <td>2</td>
      <td>maskers</td>
      <td>0.572266</td>
      <td>660</td>
    </tr>
    <tr>
      <th>29866</th>
      <td>2</td>
      <td>vaxxers</td>
      <td>0.557066</td>
      <td>465</td>
    </tr>
    <tr>
      <th>29867</th>
      <td>2</td>
      <td>masker</td>
      <td>0.541411</td>
      <td>177</td>
    </tr>
    <tr>
      <th>29868</th>
      <td>2</td>
      <td>vax</td>
      <td>0.522942</td>
      <td>504</td>
    </tr>
    <tr>
      <th>29869</th>
      <td>2</td>
      <td>vaxx</td>
      <td>0.508261</td>
      <td>132</td>
    </tr>
    <tr>
      <th>29870</th>
      <td>2</td>
      <td>lock</td>
      <td>0.490704</td>
      <td>9527</td>
    </tr>
  </tbody>
</table>
</div>


#### embeddings projection

```python
from scipy import spatial
```

```python
import altair as alt
```

```python
models = []
models.append({'subreddit': SUBRS[0], 'model': model_1})
models.append({'subreddit': SUBRS[1], 'model': model_2})
```

```python
def make_sem_axis(model, pole_word_1: str, pole_word_2: str):
	pole_1_vec = model_1.wv.get_vector(pole_1)
	pole_2_vec = model_1.wv.get_vector(pole_2)
	sem_axis = pole_1_vec - pole_2_vec
	return sem_axis

```

```python
def get_axis_sim(lex: str, pole_word_1: str, pole_word_2: str, model):
	sem_axis = make_sem_axis(model, pole_word_1, pole_word_2)
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
for model in models:
	print(f'{model["subreddit"]}: {get_axis_sim(lex, pole_1, pole_2, model["model"])}')
```

    Coronavirus: -0.08022712916135788
    LockdownSkepticism: -0.12990982830524445


```python
lexs = [
	'lockdown', 'lockdowns', 
	'shutdown', 'shutdowns', 
	'vaccine', 'vaccines', 
	'mask', 'masks',
	'order', 'police',
	'thing', 'tree', 'yellow', 'give'
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





<div id="altair-viz-f24e3dd75fe24eb18ab424c719bb51b9"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-f24e3dd75fe24eb18ab424c719bb51b9") {
      outputDiv = document.getElementById("altair-viz-f24e3dd75fe24eb18ab424c719bb51b9");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-c9778c8940eb52a5fea98a46ae8ba2b0"}, "mark": {"type": "line", "point": true}, "encoding": {"color": {"field": "subreddit", "type": "nominal"}, "x": {"field": "sim", "type": "quantitative"}, "y": {"field": "lex", "sort": null, "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-c9778c8940eb52a5fea98a46ae8ba2b0": [{"subreddit": "Coronavirus", "lex": "lockdown", "sim": -0.08022712916135788}, {"subreddit": "LockdownSkepticism", "lex": "lockdown", "sim": -0.12990982830524445}, {"subreddit": "Coronavirus", "lex": "lockdowns", "sim": -0.06833521276712418}, {"subreddit": "LockdownSkepticism", "lex": "lockdowns", "sim": -0.0952957272529602}, {"subreddit": "Coronavirus", "lex": "shutdown", "sim": -0.11067618429660797}, {"subreddit": "LockdownSkepticism", "lex": "shutdown", "sim": -0.14383257925510406}, {"subreddit": "Coronavirus", "lex": "shutdowns", "sim": -0.07748658210039139}, {"subreddit": "LockdownSkepticism", "lex": "shutdowns", "sim": -0.09972812980413437}, {"subreddit": "Coronavirus", "lex": "vaccine", "sim": -0.008722450584173203}, {"subreddit": "LockdownSkepticism", "lex": "vaccine", "sim": -0.008453463204205036}, {"subreddit": "Coronavirus", "lex": "vaccines", "sim": 0.030203502625226974}, {"subreddit": "LockdownSkepticism", "lex": "vaccines", "sim": 0.021412456408143044}, {"subreddit": "Coronavirus", "lex": "mask", "sim": 0.05741423740983009}, {"subreddit": "LockdownSkepticism", "lex": "mask", "sim": 0.03356524184346199}, {"subreddit": "Coronavirus", "lex": "masks", "sim": 0.04498105123639107}, {"subreddit": "LockdownSkepticism", "lex": "masks", "sim": 0.025375308468937874}, {"subreddit": "Coronavirus", "lex": "order", "sim": 0.0858086347579956}, {"subreddit": "LockdownSkepticism", "lex": "order", "sim": 0.0058347503654658794}, {"subreddit": "Coronavirus", "lex": "police", "sim": 0.03616446256637573}, {"subreddit": "LockdownSkepticism", "lex": "police", "sim": -0.022389892488718033}, {"subreddit": "Coronavirus", "lex": "thing", "sim": -0.08811882138252258}, {"subreddit": "LockdownSkepticism", "lex": "thing", "sim": -0.10003308206796646}, {"subreddit": "Coronavirus", "lex": "tree", "sim": -0.01825425587594509}, {"subreddit": "LockdownSkepticism", "lex": "tree", "sim": -0.01272513810545206}, {"subreddit": "Coronavirus", "lex": "yellow", "sim": -0.13428929448127747}, {"subreddit": "LockdownSkepticism", "lex": "yellow", "sim": -0.15483997762203217}, {"subreddit": "Coronavirus", "lex": "give", "sim": 0.11482276022434235}, {"subreddit": "LockdownSkepticism", "lex": "give", "sim": 0.0988151878118515}]}}, {"mode": "vega-lite"});
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
