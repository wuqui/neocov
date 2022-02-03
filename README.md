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
k = 20
freq_min = 100

sem_change_cands = (distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')
    .query('lex.str.isalpha() == True')\
    .query('lex.str.len() > 3')\
    .nlargest(k, 'dist_sem')
    .reset_index(drop=True)
)

sem_change_cands
```

```python
sem_change_cands_out = sem_change_cands\
    .nlargest(100, 'dist_sem')\
    .assign(index_1 = lambda df: df.index + 1)\
    .assign(dist_sem = lambda df: df['dist_sem'].round(2))\
    .assign(dist_sem = lambda df: df['dist_sem'].apply('{:.2f}'.format))\
    .rename({'index_1': '', 'lex': 'Lexeme', 'dist_sem': 'SemDist'}, axis=1)
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
LEX_NBS = 'chaz'
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
