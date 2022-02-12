# NeoCov
> Semantic change and social semantic variation of Covid-related English neologisms on Reddit.


```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


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
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 100)
import altair as alt
from gensim.models import Word2Vec
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
LEX_NBS = 'lockdowns'
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

#### get subreddit counts

```python
subr_counts = get_subr_counts(comments)
```

```python
subr_counts_plt = plot_subr_counts(subr_counts, k=15)
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
model_names = ['Coronavirus', 'conspiracy']
# model_names = ['Coronavirus', 'LockdownSkepticism']
```

```python
models = [dict() for name in model_names]
for i, model in enumerate(models):
	model['name'] = model_names[i]
	model['path'] = f'../out/models/{model["name"]}.model'
	model['model'] = Word2Vec.load(model['path'])

models
```




    [{'name': 'Coronavirus',
      'path': '../out/models/Coronavirus.model',
      'model': <gensim.models.word2vec.Word2Vec at 0x10d0575e0>},
     {'name': 'conspiracy',
      'path': '../out/models/conspiracy.model',
      'model': <gensim.models.word2vec.Word2Vec at 0x16f08cb20>}]



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





    <gensim.models.word2vec.Word2Vec at 0x16f08cb20>



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
models_vocab.to_csv(f'../out/vocabs/vocab_{models[0]["name"]}--{models[1]["name"]}.csv', index=False)
```

### Measure distances

```python
distances = measure_distances(models[0]['model'], models[1]['model'])
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
LEX_NBS = 'plandemic'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=10,
    model_1=models[0]['model'], 
    model_2=models[1]['model'],
    k=10
)

display(
    nbs_model_1,
    nbs_model_2
)
```

#### biggest discrepancies in nearest neighbours for target lexemes

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex='vaccine', 
    freq_min=150,
    model_1=models[0]['model'], 
    model_2=models[1]['model'],
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

### Project embeddings into subspaces

```python
lexs = [
	'regulations', 'politics',
	'government', 'mandate', 
	'science', 'research',
	'shutdown', 'shutdowns', 
	'lockdown', 'lockdowns', 
	'vaccine', 'vaccines', 
	'mask', 'masks',
	]
```

#### _good_ vs _bad_

```python
pole_words = ['good', 'bad']
```

```python
proj_sims = get_axis_sims(lexs, models, pole_words, k=10)

proj_sims_chart = alt.Chart(proj_sims).mark_line(point=True).encode(
	x='sim',
	y=alt.Y('lex', sort=None),
	color='subreddit'
)

proj_sims_chart

```





<div id="altair-viz-cf8656a131534590ab7aadc33971e29a"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-cf8656a131534590ab7aadc33971e29a") {
      outputDiv = document.getElementById("altair-viz-cf8656a131534590ab7aadc33971e29a");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-42999d1694d55e2ed0bb1c28bd6bb014"}, "mark": {"type": "line", "point": true}, "encoding": {"color": {"field": "subreddit", "type": "nominal"}, "x": {"field": "sim", "type": "quantitative"}, "y": {"field": "lex", "sort": null, "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-42999d1694d55e2ed0bb1c28bd6bb014": [{"subreddit": "Coronavirus", "lex": "regulations", "sim": -0.003947508055716753}, {"subreddit": "conspiracy", "lex": "regulations", "sim": -0.011603784747421741}, {"subreddit": "Coronavirus", "lex": "politics", "sim": -0.03446148335933685}, {"subreddit": "conspiracy", "lex": "politics", "sim": -0.02959168516099453}, {"subreddit": "Coronavirus", "lex": "government", "sim": 0.038461834192276}, {"subreddit": "conspiracy", "lex": "government", "sim": -0.11527542024850845}, {"subreddit": "Coronavirus", "lex": "mandate", "sim": 0.050309017300605774}, {"subreddit": "conspiracy", "lex": "mandate", "sim": -0.055884070694446564}, {"subreddit": "Coronavirus", "lex": "science", "sim": -0.04062918573617935}, {"subreddit": "conspiracy", "lex": "science", "sim": -0.08194876462221146}, {"subreddit": "Coronavirus", "lex": "research", "sim": 0.1142130121588707}, {"subreddit": "conspiracy", "lex": "research", "sim": 0.016525449231266975}, {"subreddit": "Coronavirus", "lex": "shutdown", "sim": -0.0030814199708402157}, {"subreddit": "conspiracy", "lex": "shutdown", "sim": -0.06672345101833344}, {"subreddit": "Coronavirus", "lex": "shutdowns", "sim": -0.061103351414203644}, {"subreddit": "conspiracy", "lex": "shutdowns", "sim": -0.12201403826475143}, {"subreddit": "Coronavirus", "lex": "lockdown", "sim": -0.020428422838449478}, {"subreddit": "conspiracy", "lex": "lockdown", "sim": -0.08437265455722809}, {"subreddit": "Coronavirus", "lex": "lockdowns", "sim": -0.03195760399103165}, {"subreddit": "conspiracy", "lex": "lockdowns", "sim": -0.13521099090576172}, {"subreddit": "Coronavirus", "lex": "vaccine", "sim": 0.027036398649215698}, {"subreddit": "conspiracy", "lex": "vaccine", "sim": -0.08384035527706146}, {"subreddit": "Coronavirus", "lex": "vaccines", "sim": 0.02113475278019905}, {"subreddit": "conspiracy", "lex": "vaccines", "sim": -0.10828987509012222}, {"subreddit": "Coronavirus", "lex": "mask", "sim": 0.02409209869801998}, {"subreddit": "conspiracy", "lex": "mask", "sim": -0.10277207940816879}, {"subreddit": "Coronavirus", "lex": "masks", "sim": 0.042272794991731644}, {"subreddit": "conspiracy", "lex": "masks", "sim": -0.1298701912164688}]}}, {"mode": "vega-lite"});
</script>



#### _objective_ vs _subjective_

```python
pole_words = ['objective', 'subjective']
```

```python
proj_sims = get_axis_sims(lexs, models, pole_words, k=10)

proj_sims_chart = alt.Chart(proj_sims).mark_line(point=True).encode(
	x='sim',
	y=alt.Y('lex', sort=None),
	color='subreddit'
)

proj_sims_chart

```





<div id="altair-viz-dbae6d5e5ca144ff8e7c1ddd3990e69b"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-dbae6d5e5ca144ff8e7c1ddd3990e69b") {
      outputDiv = document.getElementById("altair-viz-dbae6d5e5ca144ff8e7c1ddd3990e69b");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-e55b97b1001d1e7c73b505459fc4c90d"}, "mark": {"type": "line", "point": true}, "encoding": {"color": {"field": "subreddit", "type": "nominal"}, "x": {"field": "sim", "type": "quantitative"}, "y": {"field": "lex", "sort": null, "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-e55b97b1001d1e7c73b505459fc4c90d": [{"subreddit": "Coronavirus", "lex": "regulations", "sim": -0.03523268178105354}, {"subreddit": "conspiracy", "lex": "regulations", "sim": -0.08504713326692581}, {"subreddit": "Coronavirus", "lex": "politics", "sim": 0.028386631980538368}, {"subreddit": "conspiracy", "lex": "politics", "sim": 0.04286649823188782}, {"subreddit": "Coronavirus", "lex": "government", "sim": 0.06713764369487762}, {"subreddit": "conspiracy", "lex": "government", "sim": 0.01848958246409893}, {"subreddit": "Coronavirus", "lex": "mandate", "sim": 0.056924622505903244}, {"subreddit": "conspiracy", "lex": "mandate", "sim": 0.019045785069465637}, {"subreddit": "Coronavirus", "lex": "science", "sim": 0.11393926292657852}, {"subreddit": "conspiracy", "lex": "science", "sim": 0.06988415122032166}, {"subreddit": "Coronavirus", "lex": "research", "sim": 0.10565640777349472}, {"subreddit": "conspiracy", "lex": "research", "sim": 0.12913458049297333}, {"subreddit": "Coronavirus", "lex": "shutdown", "sim": 0.032770637422800064}, {"subreddit": "conspiracy", "lex": "shutdown", "sim": 0.011807557195425034}, {"subreddit": "Coronavirus", "lex": "shutdowns", "sim": -0.04867933690547943}, {"subreddit": "conspiracy", "lex": "shutdowns", "sim": -0.06140519678592682}, {"subreddit": "Coronavirus", "lex": "lockdown", "sim": -0.005724509246647358}, {"subreddit": "conspiracy", "lex": "lockdown", "sim": -0.0689033567905426}, {"subreddit": "Coronavirus", "lex": "lockdowns", "sim": -0.006858906242996454}, {"subreddit": "conspiracy", "lex": "lockdowns", "sim": -0.09060239046812057}, {"subreddit": "Coronavirus", "lex": "vaccine", "sim": 0.08758274465799332}, {"subreddit": "conspiracy", "lex": "vaccine", "sim": 0.020923763513565063}, {"subreddit": "Coronavirus", "lex": "vaccines", "sim": -0.005105141084641218}, {"subreddit": "conspiracy", "lex": "vaccines", "sim": -0.03825485333800316}, {"subreddit": "Coronavirus", "lex": "mask", "sim": -0.027438262477517128}, {"subreddit": "conspiracy", "lex": "mask", "sim": -0.10273142904043198}, {"subreddit": "Coronavirus", "lex": "masks", "sim": 0.0008544788579456508}, {"subreddit": "conspiracy", "lex": "masks", "sim": -0.0715503916144371}]}}, {"mode": "vega-lite"});
</script>



### Plot embedding space

```python
lex_vecs = []
for lex in lexs:
	for model in models:
		lex_d = {}
		lex_d['lex'] = lex
		lex_d['subreddit'] = model['subreddit']
		lex_d['vec'] = model['model'].wv.get_vector(lex)
		lex_vecs.append(lex_d)
```

```python
lex = 'lockdown'
lex_vecs = []

for model in models:
	lex_d = {}
	lex_d['lex'] = lex
	lex_d['type'] = 'center'
	lex_d['subreddit'] = model['subreddit']
	lex_d['vec'] = model['model'].wv.get_vector(lex)
	lex_vecs.append(lex_d)
	for nb, sim in model['model'].wv.most_similar(lex, topn=50):
		lex_d = {}
		lex_d['lex'] = nb
		lex_d['type'] = 'nb'
		lex_d['subreddit'] = model['subreddit']
		lex_d['vec'] =  model['model'].wv.get_vector(nb)
		lex_vecs.append(lex_d)
```

```python
vecs_df = pd.DataFrame(lex_vecs)
vecs_df
```

```python
from sklearn.manifold import TSNE
```

```python

Y_tsne = TSNE(
    perplexity=70,
    method='exact',
    init='pca',
    verbose=True
    )\
    .fit_transform(list(vecs_df['vec']))

vecs_df['x_tsne'] = Y_tsne[:, [0]]
vecs_df['y_tsne'] = Y_tsne[:, [1]]

```

```python
brush = alt.selection(
    type="interval",
    on="[mousedown[event.altKey], mouseup] > mousemove",
    translate="[mousedown[event.altKey], mouseup] > mousemove!",
    zoom="wheel![event.altKey]",
)

interaction = alt.selection(
    type="interval",
    bind="scales",
    on="[mousedown[!event.altKey], mouseup] > mousemove",
    translate="[mousedown[!event.altKey], mouseup] > mousemove!",
    zoom="wheel![!event.altKey]",
)

chart = (alt.Chart(vecs_df).mark_text(point=True).encode(
	x = 'x_tsne',
	y = 'y_tsne',
	text = 'lex',
	size = alt.condition("datum.type == 'center'", alt.value(25), alt.value(10)),
	color = alt.condition(brush, 'subreddit', alt.value('lightgray')),
	column = 'subreddit'
	)
	.properties(title=f"Social semantic variation for the word '{lex}'.")
	.add_selection(brush, interaction)
)

chart
```

Link to interactive chart: https://wuqui.github.io/neocov/#Plot-embedding-space.

Press and hold the <kbd>alt</kbd> key to select regions of the semantic space.
