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
```

```python
DATA_DIR = '../data/'
COMMENTS_DIAC_DIR = f'{DATA_DIR}comments/by_date/'
OUT_DIR = '../out/'
```

## Variables

## Semantic change

### Read data

#### Get file paths

```python
YEAR = '2019'
```

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
comments = clean_comments(comments)
```

### Train models

#### Create corpus

```python
corpus = Corpus(comments['body'])
```

#### Train model

```python
%%time
model = train_model(corpus)
```

```python
len(model.wv.key_to_index)
```

#### Save model

```python
model.save(f'{OUT_DIR}models/{YEAR}.model')
```

#### Load models

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

```python
distances\
    .sort_values('dist_sem', ascending=False)

```

TODO: filter by true type frequency; `Gensim`'s type frequency seems incorrect; it probably reflects frequency ranks instead of total counts.

```python
def get_sem_change_cands(distances, k=10, freq_min=1):
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
k = 20
freq_min = 1_000

sem_change_cands = distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')\
    .query('lex.str.isalpha() == True')\
    .query('lex.str.len() > 3')\
    .nlargest(k, 'dist_sem')\
    .reset_index(drop=True)

sem_change_cands
```

```python
sem_change_cands_out = sem_change_cands\
    .nlargest(100, 'dist_sem')\
    .assign(index_1 = lambda df: df.index + 1)\
    .assign(dist_sem = lambda df: df['dist_sem'].round(2))\
    .assign(dist_sem = lambda df: df['dist_sem'].apply('{:.2f}'.format))\
    .rename({'index_1': '', 'lex': 'Lexeme', 'dist_sem': 'SemDist'}, axis=1)

sem_change_cands_out.head(20)
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
LEX_NBS = 'distancing'
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

```python
nbs_model_1.to_csv(f'{OUT_DIR}neighbours/{LEX_NBS}_2019.csv')
nbs_model_2.to_csv(f'{OUT_DIR}neighbours/{LEX_NBS}_2020.csv')
```

## Social semantic variation

### Inspect subreddits

#### read comments

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

    CPU times: user 48.6 s, sys: 6.54 s, total: 55.1 s
    Wall time: 56 s





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

#### get subreddit counts

```python
subr_counts = get_subr_counts(comments)
```

```python
subr_counts_plt = plot_subr_counts(subr_counts, k=20)
subr_counts_plt
```





<div id="altair-viz-28f42ba5ccb5467bbf5b008ccdfa8388"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-28f42ba5ccb5467bbf5b008ccdfa8388") {
      outputDiv = document.getElementById("altair-viz-28f42ba5ccb5467bbf5b008ccdfa8388");
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
subr_counts_plt.save(f'{OUT_DIR}subr_counts.png', scale_factor=2.0)
```

```python
comments\
    .query('subreddit == "AskReddit"')\
     .sample(10)
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
      <th>2720703</th>
      <td>Blueowl789</td>
      <td>oh yeah? quote it</td>
      <td>2019-12-07 22:45:53</td>
      <td>fa0ql3q</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>6612440</th>
      <td>ghostoflops</td>
      <td>Lurking for jerking off, participating for con...</td>
      <td>2019-09-01 21:07:19</td>
      <td>eyrlxlg</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>8541016</th>
      <td>WhenAllElseFail</td>
      <td>so i put my hands up, they're playing my song,...</td>
      <td>2019-08-14 21:46:21</td>
      <td>eww2p0q</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>6482914</th>
      <td>Tyr_ranical</td>
      <td>Wait is this reply about stuff here, or a thin...</td>
      <td>2019-03-19 22:29:49</td>
      <td>eiwy4fz</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>599903</th>
      <td>JazzUnlikeTheCaroot</td>
      <td>You as a director 😀</td>
      <td>2019-07-14 21:59:58</td>
      <td>etsbx0o</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>6102560</th>
      <td>_luckybandit_</td>
      <td>DashieXP or Dashie games

He stopped uploading...</td>
      <td>2019-05-01 21:34:58</td>
      <td>em9fj76</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>7595345</th>
      <td>neovangelis</td>
      <td>Periscope for "wizards"</td>
      <td>2019-08-19 21:58:51</td>
      <td>exfzg04</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>2155937</th>
      <td>Byrinthion</td>
      <td>I’m not a developer haha, I’m a writer. So I d...</td>
      <td>2019-03-14 22:47:57</td>
      <td>eijpp3m</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>2698189</th>
      <td>AutoModerator</td>
      <td>**PLEASE READ THIS MESSAGE IN ITS ENTIRETY BEF...</td>
      <td>2019-12-07 22:41:05</td>
      <td>fa0q2w2</td>
      <td>AskReddit</td>
    </tr>
    <tr>
      <th>7483454</th>
      <td>neovangelis</td>
      <td>The Native Americans who came before them woul...</td>
      <td>2019-08-19 21:30:52</td>
      <td>exfwxci</td>
      <td>AskReddit</td>
    </tr>
  </tbody>
</table>
</div>


