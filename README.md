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
from pathlib import Path
import pandas as pd
pd.set_option('display.max_rows', 100)
import altair as alt
from altair_saver import save
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
      'model': <gensim.models.word2vec.Word2Vec at 0x16624e800>},
     {'name': 'conspiracy',
      'path': '../out/models/conspiracy.model',
      'model': <gensim.models.word2vec.Word2Vec at 0x17359ffd0>}]



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





    <gensim.models.word2vec.Word2Vec at 0x17359ffd0>



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

### Project embeddings on semantix axes

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
pole_words_pos = ['good', 'bad']
```

```python
proj_sims_pos = get_axis_sims(lexs, models, pole_words_pos, k=10)
```

```python
proj_sims_pos_chart = alt.Chart(proj_sims_pos).mark_line(point=True).encode(
	x=alt.X('sim', title='SemSim'),
	y=alt.Y('lex', title='', sort=None),
	color=alt.Color('subreddit', title='Community')
).properties(title=f'{pole_words_pos[0]} vs {pole_words_pos[1]}')

proj_sims_pos_chart
```

```python
proj_sims_pos_chart.save(f'../out/proj-emb_pos_{models[0]["name"]}--{models[1]["name"]}.pdf')
```

#### _objective_ vs _subjective_

```python
pole_words_subj = ['objective', 'subjective']
```

```python
proj_sims_subj = get_axis_sims(lexs, models, pole_words_subj, k=10)
```

```python
proj_sims_subj_chart = alt.Chart(proj_sims_subj).mark_line(point=True).encode(
	x=alt.X('sim', title='SemSim'),
	y=alt.Y('lex', title='', sort=None),
	color=alt.Color('subreddit', title='Community')
).properties(title=f'{pole_words_pos[0]} vs {pole_words_pos[1]}')

proj_sims_subj_chart
```

```python
proj_sims_subj_chart.save(f'../out/proj-emb_subj_{models[0]["name"]}--{models[1]["name"]}.pdf')
```

### Plot embedding space

```python
lex = 'vaccine'
```

```python
#data
nbs_vecs = pd.concat([get_nbs_vecs(lex, model) for model in models])
```

```python
#data
nbs_vecs = dim_red_nbs_vecs(nbs_vecs)
```

    /Users/quirin/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:790: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      warnings.warn(
    /Users/quirin/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/sklearn/manifold/_t_sne.py:982: FutureWarning: The PCA initialization in TSNE will change to have the standard deviation of PC1 equal to 1e-4 in 1.2. This will ensure better convergence.
      warnings.warn(


    [t-SNE] Computing pairwise distances...
    [t-SNE] Computed conditional probabilities for sample 102 / 102
    [t-SNE] Mean sigma: 11.515156
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 48.863581
    [t-SNE] KL divergence after 550 iterations: 0.069542


```python
#data
nbs_vecs_chart = plot_nbs_vecs(lex, nbs_vecs)
nbs_vecs_chart
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/gp/dw55jb3d3gl6jn22rscvxjm40000gn/T/ipykernel_23795/1278045993.py in <module>
          1 #data
    ----> 2 nbs_vecs_chart = plot_nbs_vecs(nbs_vecs)
          3 nbs_vecs_chart


    ~/promo/NeoCov/neocov/neocov/type_emb.py in plot_nbs_vecs(nbs_vecs, perplexity)
        276                 column = 'subreddit'
        277 		)
    --> 278                 .properties(title=f"Social semantic variation for the word '{lex}'.")
        279                 .add_selection(brush, interaction)
        280 	)


    NameError: name 'lex' is not defined


```python
#data
nbs_vecs_chart.save(f'../out/map-sem-space_{lex}_{models[0]["name"]}--{models[1]["name"]}.pdf')
nbs_vecs_chart.save(f'../out/map-sem-space_{lex}_{models[0]["name"]}--{models[1]["name"]}.html')
```

Link to interactive chart: https://wuqui.github.io/neocov/#Plot-embedding-space.

Press and hold the <kbd>alt</kbd> key to select regions of the semantic space.
