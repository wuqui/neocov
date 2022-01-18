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

    190756 190756
    190756 190756





    <gensim.models.word2vec.Word2Vec at 0x184aee710>



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
      <th>182174</th>
      <td>financiados</td>
      <td>1.270406</td>
      <td>7</td>
      <td>9</td>
    </tr>
    <tr>
      <th>164003</th>
      <td>______________________________________________...</td>
      <td>1.257892</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>181373</th>
      <td>2ffireemblem</td>
      <td>1.247719</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>190665</th>
      <td>obedece</td>
      <td>1.239514</td>
      <td>7</td>
      <td>8</td>
    </tr>
    <tr>
      <th>126286</th>
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
      <th>174923</th>
      <td>ppx_yo_dt_b_asin_title_o09_s00</td>
      <td>0.025620</td>
      <td>8</td>
      <td>9</td>
    </tr>
    <tr>
      <th>46509</th>
      <td>imagestabilization</td>
      <td>0.025614</td>
      <td>85</td>
      <td>93</td>
    </tr>
    <tr>
      <th>144055</th>
      <td>ppx_yo_dt_b_asin_title_o03_s00</td>
      <td>0.018814</td>
      <td>11</td>
      <td>13</td>
    </tr>
    <tr>
      <th>68400</th>
      <td>u5e9htr</td>
      <td>0.012333</td>
      <td>42</td>
      <td>46</td>
    </tr>
  </tbody>
</table>
<p>190756 rows × 4 columns</p>
</div>



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
freq_min = 100

sem_change_cands = distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')\
    .query('lex.str.isalpha() == True')\
    .query('lex.str.len() > 3')\
    .nlargest(k, 'dist_sem')\
    .reset_index(drop=True)

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
      <td>maskless</td>
      <td>1.100272</td>
      <td>118</td>
      <td>127</td>
    </tr>
    <tr>
      <th>1</th>
      <td>lockdowns</td>
      <td>1.070362</td>
      <td>940</td>
      <td>991</td>
    </tr>
    <tr>
      <th>2</th>
      <td>sunsetting</td>
      <td>1.039729</td>
      <td>111</td>
      <td>120</td>
    </tr>
    <tr>
      <th>3</th>
      <td>chaz</td>
      <td>1.010383</td>
      <td>190</td>
      <td>202</td>
    </tr>
    <tr>
      <th>4</th>
      <td>childe</td>
      <td>0.957373</td>
      <td>209</td>
      <td>222</td>
    </tr>
    <tr>
      <th>5</th>
      <td>cerb</td>
      <td>0.957321</td>
      <td>315</td>
      <td>333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>megalodon</td>
      <td>0.937414</td>
      <td>752</td>
      <td>792</td>
    </tr>
    <tr>
      <th>7</th>
      <td>spreader</td>
      <td>0.932299</td>
      <td>164</td>
      <td>175</td>
    </tr>
    <tr>
      <th>8</th>
      <td>corona</td>
      <td>0.927504</td>
      <td>3553</td>
      <td>3684</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ventilators</td>
      <td>0.925241</td>
      <td>384</td>
      <td>405</td>
    </tr>
    <tr>
      <th>10</th>
      <td>woul</td>
      <td>0.924612</td>
      <td>197</td>
      <td>210</td>
    </tr>
    <tr>
      <th>11</th>
      <td>rona</td>
      <td>0.922876</td>
      <td>410</td>
      <td>433</td>
    </tr>
    <tr>
      <th>12</th>
      <td>revamping</td>
      <td>0.914010</td>
      <td>129</td>
      <td>139</td>
    </tr>
    <tr>
      <th>13</th>
      <td>pandemic</td>
      <td>0.912615</td>
      <td>9504</td>
      <td>9957</td>
    </tr>
    <tr>
      <th>14</th>
      <td>snapchatting</td>
      <td>0.912304</td>
      <td>2262</td>
      <td>2345</td>
    </tr>
    <tr>
      <th>15</th>
      <td>diys</td>
      <td>0.910311</td>
      <td>492</td>
      <td>519</td>
    </tr>
    <tr>
      <th>16</th>
      <td>rittenhouse</td>
      <td>0.904660</td>
      <td>181</td>
      <td>193</td>
    </tr>
    <tr>
      <th>17</th>
      <td>hyperlinked</td>
      <td>0.904281</td>
      <td>511</td>
      <td>538</td>
    </tr>
    <tr>
      <th>18</th>
      <td>pandemics</td>
      <td>0.902996</td>
      <td>296</td>
      <td>311</td>
    </tr>
    <tr>
      <th>19</th>
      <td>rpan</td>
      <td>0.891097</td>
      <td>542</td>
      <td>573</td>
    </tr>
  </tbody>
</table>
</div>



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
LEX_NBS = 'lockdown'
```

```python
nbs_model_1, nbs_model_2 = get_nearest_neighbours_models(
    lex=LEX_NBS, 
    freq_min=1,
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
      <td>perimeter</td>
      <td>0.486453</td>
      <td>530</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>rotation</td>
      <td>0.438531</td>
      <td>3259</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>prowl</td>
      <td>0.422064</td>
      <td>79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>standby</td>
      <td>0.421939</td>
      <td>374</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>spotter</td>
      <td>0.418285</td>
      <td>171</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>turret</td>
      <td>0.412857</td>
      <td>1260</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>hof</td>
      <td>0.409758</td>
      <td>797</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>cqc</td>
      <td>0.408991</td>
      <td>198</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>galleon</td>
      <td>0.406334</td>
      <td>124</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>frontline</td>
      <td>0.403402</td>
      <td>678</td>
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
      <td>lockdowns</td>
      <td>0.764840</td>
      <td>991</td>
    </tr>
    <tr>
      <th>100001</th>
      <td>2</td>
      <td>quarantine</td>
      <td>0.761415</td>
      <td>5225</td>
    </tr>
    <tr>
      <th>100002</th>
      <td>2</td>
      <td>pandemic</td>
      <td>0.717629</td>
      <td>9957</td>
    </tr>
    <tr>
      <th>100003</th>
      <td>2</td>
      <td>shutdown</td>
      <td>0.714777</td>
      <td>1905</td>
    </tr>
    <tr>
      <th>100004</th>
      <td>2</td>
      <td>shutdowns</td>
      <td>0.636877</td>
      <td>226</td>
    </tr>
    <tr>
      <th>100005</th>
      <td>2</td>
      <td>quarantines</td>
      <td>0.601028</td>
      <td>160</td>
    </tr>
    <tr>
      <th>100006</th>
      <td>2</td>
      <td>corona</td>
      <td>0.579884</td>
      <td>3684</td>
    </tr>
    <tr>
      <th>100007</th>
      <td>2</td>
      <td>reopening</td>
      <td>0.565140</td>
      <td>757</td>
    </tr>
    <tr>
      <th>100008</th>
      <td>2</td>
      <td>outbreak</td>
      <td>0.557197</td>
      <td>2213</td>
    </tr>
    <tr>
      <th>100009</th>
      <td>2</td>
      <td>virus</td>
      <td>0.550216</td>
      <td>14315</td>
    </tr>
  </tbody>
</table>
</div>


```python
nbs_model_1.to_csv(f'{OUT_DIR}neighbours/{LEX_NBS}_2019.csv')
nbs_model_2.to_csv(f'{OUT_DIR}neighbours/{LEX_NBS}_2020.csv')
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

    CPU times: user 34.2 s, sys: 3.62 s, total: 37.8 s
    Wall time: 37.9 s





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
      <td>Strep swab and culture negative, I’m sure? Cou...</td>
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
<p>3800765 rows × 5 columns</p>
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





<div id="altair-viz-be48b7c570e146cbb8e25c8f162c7758"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-be48b7c570e146cbb8e25c8f162c7758") {
      outputDiv = document.getElementById("altair-viz-be48b7c570e146cbb8e25c8f162c7758");
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
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "data": {"name": "data-4458ddb2ac6f25acdcc8df0cbd1ee925"}, "mark": "bar", "encoding": {"x": {"field": "comments_num", "type": "quantitative"}, "y": {"field": "subreddit", "sort": "-x", "type": "nominal"}}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-4458ddb2ac6f25acdcc8df0cbd1ee925": [{"subreddit": "Coronavirus", "comments_num": 173397}, {"subreddit": "AskReddit", "comments_num": 142358}, {"subreddit": "politics", "comments_num": 136270}, {"subreddit": "worldnews", "comments_num": 87336}, {"subreddit": "news", "comments_num": 55561}, {"subreddit": "conspiracy", "comments_num": 52780}, {"subreddit": "wallstreetbets", "comments_num": 40424}, {"subreddit": "DDnews", "comments_num": 37062}, {"subreddit": "AmItheAsshole", "comments_num": 35861}, {"subreddit": "pics", "comments_num": 27597}, {"subreddit": "ukpolitics", "comments_num": 25970}, {"subreddit": "LockdownSkepticism", "comments_num": 22906}, {"subreddit": "PublicFreakout", "comments_num": 22891}, {"subreddit": "canada", "comments_num": 22754}, {"subreddit": "Conservative", "comments_num": 21673}, {"subreddit": "nfl", "comments_num": 20222}, {"subreddit": "memes", "comments_num": 18200}, {"subreddit": "COVID19", "comments_num": 17570}, {"subreddit": "ontario", "comments_num": 16386}, {"subreddit": "relationship_advice", "comments_num": 16143}]}}, {"mode": "vega-lite"});
</script>



```python
subr_counts_plt.save(f'{OUT_DIR}subr_counts.png', scale_factor=2.0)
```

## Train models

```python
COMMENTS_DIR_SUBR = '../data/comments/subr/'
```

```python
SUBR = 'Coronavirus'
```

```python
fpaths = get_comments_paths_subr(COMMENTS_DIR_SUBR, SUBR)
```

```python
%%time
comments = read_comm_csvs(fpaths)
```

    CPU times: user 24 s, sys: 2.71 s, total: 26.7 s
    Wall time: 26.8 s


```python
%%time
comments = clean_comments(comments)
```

    conv_to_lowerc       (4121144, 5) 0:00:03.492902      
    rm_punct             (4121144, 5) 0:00:26.037508      
    tokenize             (4121144, 5) 0:03:10.999946      
    rem_short_comments   (3462555, 5) 0:00:58.288521      
    CPU times: user 1min 4s, sys: 1min 29s, total: 2min 33s
    Wall time: 4min 57s


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
      <td>bikbar1</td>
      <td>[gt, but, it, s, still, impossible, to, hide, ...</td>
      <td>2020-09-06 10:11:45</td>
      <td>g47wejw</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>1</th>
      <td>righteousprovidence</td>
      <td>[my, guess, is, americans, don, t, see, weldin...</td>
      <td>2020-09-06 10:13:17</td>
      <td>g47whmx</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>2</th>
      <td>liriodendron1</td>
      <td>[i, dont, want, compensation, i, want, it, to,...</td>
      <td>2020-09-06 10:13:27</td>
      <td>g47whzg</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ArbitraryBaker</td>
      <td>[except, the, testing, is, flawed, too, have, ...</td>
      <td>2020-09-06 10:14:02</td>
      <td>g47wj5l</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mogambis</td>
      <td>[little, did, he, know, she, is, an, it]</td>
      <td>2020-09-06 10:15:33</td>
      <td>g47wm3e</td>
      <td>Coronavirus</td>
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
      <th>4121139</th>
      <td>LouQuacious</td>
      <td>[it, s, the, 21st, century, no, excuse, for, n...</td>
      <td>2020-12-14 22:59:46</td>
      <td>gfv1k7s</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>4121140</th>
      <td>immibis</td>
      <td>[covid, has, a, 1, in, 500, side, effect, of, ...</td>
      <td>2020-12-14 22:59:50</td>
      <td>gfv1kho</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>4121141</th>
      <td>starlordbg</td>
      <td>[i, would, personally, wait, a, few, years, to...</td>
      <td>2020-12-14 22:59:53</td>
      <td>gfv1kqa</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>4121142</th>
      <td>ihadanamebutforgot</td>
      <td>[cool, dude, lemme, know, when, science, eradi...</td>
      <td>2020-12-14 22:59:57</td>
      <td>gfv1kzb</td>
      <td>Coronavirus</td>
    </tr>
    <tr>
      <th>4121143</th>
      <td>Iknowwecanmakeit</td>
      <td>[hey, someone, has, to, represent, businesses, s]</td>
      <td>2020-12-14 22:59:57</td>
      <td>gfv1l1e</td>
      <td>Coronavirus</td>
    </tr>
  </tbody>
</table>
<p>3462555 rows × 5 columns</p>
</div>



```python
corpus = Corpus(comments['body'])
```

```python
%%time
model = train_model(corpus)
```

    CPU times: user 27min 59s, sys: 45.9 s, total: 28min 45s
    Wall time: 6min 48s


```python
len(model.wv.key_to_index)
```




    94816



```python
model.save(f'{OUT_DIR}models/{SUBR}.model')
```

## Load models

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





    <gensim.models.word2vec.Word2Vec at 0x1c5394850>



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

```python
freq_min = 100

distances\
    .query('freq_1 > @freq_min and freq_2 > @freq_min')\
    .sort_values('dist_sem', ascending=False)\
    .head(20)

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
      <th>21288</th>
      <td>ptb</td>
      <td>1.144010</td>
      <td>103</td>
      <td>144</td>
    </tr>
    <tr>
      <th>270</th>
      <td>sticky</td>
      <td>1.035551</td>
      <td>76776</td>
      <td>69776</td>
    </tr>
    <tr>
      <th>18320</th>
      <td>refraction</td>
      <td>1.021251</td>
      <td>142</td>
      <td>196</td>
    </tr>
    <tr>
      <th>9409</th>
      <td>accumulative</td>
      <td>1.011545</td>
      <td>539</td>
      <td>667</td>
    </tr>
    <tr>
      <th>1262</th>
      <td>pms</td>
      <td>1.010546</td>
      <td>11535</td>
      <td>11405</td>
    </tr>
    <tr>
      <th>3996</th>
      <td>soliciting</td>
      <td>0.989504</td>
      <td>2233</td>
      <td>2472</td>
    </tr>
    <tr>
      <th>4719</th>
      <td>resubmit</td>
      <td>0.944866</td>
      <td>1763</td>
      <td>1928</td>
    </tr>
    <tr>
      <th>895</th>
      <td>ss</td>
      <td>0.944818</td>
      <td>18100</td>
      <td>17373</td>
    </tr>
    <tr>
      <th>16111</th>
      <td>ets</td>
      <td>0.933242</td>
      <td>184</td>
      <td>252</td>
    </tr>
    <tr>
      <th>17845</th>
      <td>waiters</td>
      <td>0.929376</td>
      <td>150</td>
      <td>206</td>
    </tr>
    <tr>
      <th>257</th>
      <td>chain</td>
      <td>0.929052</td>
      <td>79084</td>
      <td>73908</td>
    </tr>
    <tr>
      <th>19967</th>
      <td>curiously</td>
      <td>0.928779</td>
      <td>119</td>
      <td>164</td>
    </tr>
    <tr>
      <th>1210</th>
      <td>subsequently</td>
      <td>0.924492</td>
      <td>12174</td>
      <td>11956</td>
    </tr>
    <tr>
      <th>16368</th>
      <td>formats</td>
      <td>0.921073</td>
      <td>178</td>
      <td>244</td>
    </tr>
    <tr>
      <th>13569</th>
      <td>anons</td>
      <td>0.920570</td>
      <td>259</td>
      <td>351</td>
    </tr>
    <tr>
      <th>20911</th>
      <td>blacklivesmatter</td>
      <td>0.918460</td>
      <td>108</td>
      <td>150</td>
    </tr>
    <tr>
      <th>140</th>
      <td>meta</td>
      <td>0.917502</td>
      <td>144695</td>
      <td>156993</td>
    </tr>
    <tr>
      <th>19917</th>
      <td>redpill</td>
      <td>0.914164</td>
      <td>119</td>
      <td>165</td>
    </tr>
    <tr>
      <th>15058</th>
      <td>derek</td>
      <td>0.913568</td>
      <td>211</td>
      <td>288</td>
    </tr>
    <tr>
      <th>14487</th>
      <td>ic</td>
      <td>0.902983</td>
      <td>228</td>
      <td>310</td>
    </tr>
  </tbody>
</table>
</div>



```python
LEX = 'incivility'
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
      <td>submission</td>
      <td>0.441016</td>
      <td>50354</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>sub</td>
      <td>0.373823</td>
      <td>74717</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>harping</td>
      <td>0.301673</td>
      <td>125</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>hellbent</td>
      <td>0.296312</td>
      <td>112</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>tack</td>
      <td>0.292103</td>
      <td>155</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>predicated</td>
      <td>0.287444</td>
      <td>185</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>fixated</td>
      <td>0.283126</td>
      <td>170</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>par</td>
      <td>0.280482</td>
      <td>1222</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>standby</td>
      <td>0.272287</td>
      <td>127</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>drags</td>
      <td>0.271937</td>
      <td>148</td>
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
      <th>21216</th>
      <td>2</td>
      <td>perma</td>
      <td>0.353852</td>
      <td>205</td>
    </tr>
    <tr>
      <th>21217</th>
      <td>2</td>
      <td>mistook</td>
      <td>0.321916</td>
      <td>121</td>
    </tr>
    <tr>
      <th>21218</th>
      <td>2</td>
      <td>thanked</td>
      <td>0.311947</td>
      <td>232</td>
    </tr>
    <tr>
      <th>21219</th>
      <td>2</td>
      <td>thankyou</td>
      <td>0.306021</td>
      <td>131</td>
    </tr>
    <tr>
      <th>21220</th>
      <td>2</td>
      <td>strategist</td>
      <td>0.304107</td>
      <td>196</td>
    </tr>
    <tr>
      <th>21221</th>
      <td>2</td>
      <td>uncalled</td>
      <td>0.302803</td>
      <td>126</td>
    </tr>
    <tr>
      <th>21222</th>
      <td>2</td>
      <td>purposes</td>
      <td>0.300664</td>
      <td>2766</td>
    </tr>
    <tr>
      <th>21223</th>
      <td>2</td>
      <td>wrongthink</td>
      <td>0.288576</td>
      <td>251</td>
    </tr>
    <tr>
      <th>21224</th>
      <td>2</td>
      <td>apologizes</td>
      <td>0.284437</td>
      <td>118</td>
    </tr>
    <tr>
      <th>21225</th>
      <td>2</td>
      <td>ardern</td>
      <td>0.282216</td>
      <td>141</td>
    </tr>
  </tbody>
</table>
</div>

