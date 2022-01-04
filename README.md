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
```

```python
import pandas as pd
```

## Read data

### Get file paths

```python
COMMENTS_DIR = '../data/comments/by_date/'
```

```python
YEAR = '2019'
```

```python
comment_paths_year = get_comments_paths_year(COMMENTS_DIR, YEAR)
```

### Read comments

```python
%%time
comments = read_comm_csvs(comment_paths_year)
```

    CPU times: user 54.9 s, sys: 35.7 s, total: 1min 30s
    Wall time: 3min 38s


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



```python
comments.value_counts('subreddit')
```




    subreddit
    AskReddit             429516
    politics              146023
    memes                  99027
    teenagers              89685
    dankmemes              84107
                           ...  
    no_u                       1
    CuteBobby                  1
    no_drama                   1
    WorldBoxGodSandbox         1
    FatFurryPorn               1
    Length: 66885, dtype: int64



## Pre-process comments

### run preprocessing

```python
%%time
docs_clean = clean_docs(comments['body'])
```

    CPU times: user 2min 12s, sys: 6min 28s, total: 8min 41s
    Wall time: 36min 17s


```python
comments_sm = comments.iloc[:100]
```

```python
comments_sm.loc[:, 'body'].apply(conv_to_lowerc)
```




    0                                   username checks out
    1     if this is a dank meme, **upvote** this commen...
    2     just threaten them that you'll call the corpor...
    3                                             [removed]
    4     honestly, do you really wanna go through an en...
                                ...                        
    95    thank you! \n\ni had someone ask me in person ...
    96    people always imagine robots taking over the h...
    97                 sexy before and after!   good job...
    98                                jk, i only want frank
    99    not sure if this belongs here but here we go.\...
    Name: body, Length: 100, dtype: object



### save to disk

#### `csv`

```python
%%time
docs_clean.to_csv(f'../data/docs_clean/{YEAR}.csv', index=False)
```

    CPU times: user 51.3 s, sys: 36.7 s, total: 1min 28s
    Wall time: 3min 55s


#### `feather`

```python
docs_clean_fr = docs_clean.to_frame()
```

```python
type(docs_clean_fr.iloc[0])
```




    pandas.core.series.Series



```python
docs_clean.to_feather(f'../data/docs_clean/{YEAR}.feather')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /var/folders/gp/dw55jb3d3gl6jn22rscvxjm40000gn/T/ipykernel_56072/3858824913.py in <module>
    ----> 1 docs_clean.to_feather(f'../data/docs_clean/{YEAR}.feather')
    

    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/core/generic.py in __getattr__(self, name)
       5485         ):
       5486             return self[name]
    -> 5487         return object.__getattribute__(self, name)
       5488 
       5489     def __setattr__(self, name: str, value) -> None:


    AttributeError: 'Series' object has no attribute 'to_feather'


```python
docs_clean_feath = pd.read_feather('~/Desktop/comments.feather')
```

```python
comments
```

### load from disk

```python
%%time
docs_clean = pd.read_csv(f'../data/docs_clean/{YEAR}.csv', index_col=0, header=None)
```

    CPU times: user 21 s, sys: 1.52 s, total: 22.5 s
    Wall time: 22.9 s


```python
%%time
docs_clean = pd.read_csv(f'../data/docs_clean/{YEAR}.csv', converters={'body': pd.eval})
```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <timed exec> in <module>


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/util/_decorators.py in wrapper(*args, **kwargs)
        309                     stacklevel=stacklevel,
        310                 )
    --> 311             return func(*args, **kwargs)
        312 
        313         return wrapper


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        584     kwds.update(kwds_defaults)
        585 
    --> 586     return _read(filepath_or_buffer, kwds)
        587 
        588 


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
        486 
        487     with parser:
    --> 488         return parser.read(nrows)
        489 
        490 


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/io/parsers/readers.py in read(self, nrows)
       1045     def read(self, nrows=None):
       1046         nrows = validate_integer("nrows", nrows)
    -> 1047         index, columns, col_dict = self._engine.read(nrows)
       1048 
       1049         if index is None:


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/io/parsers/c_parser_wrapper.py in read(self, nrows)
        222         try:
        223             if self.low_memory:
    --> 224                 chunks = self._reader.read_low_memory(nrows)
        225                 # destructive to chunks
        226                 data = _concatenate_chunks(chunks)


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader.read_low_memory()


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._read_rows()


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/_libs/parsers.pyx in pandas._libs.parsers.TextReader._convert_column_data()


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/_libs/parsers.pyx in pandas._libs.parsers._apply_converter()


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/core/computation/eval.py in eval(expr, parser, engine, truediv, local_dict, global_dict, resolvers, level, target, inplace)
        335     for expr in exprs:
        336         expr = _convert_expression(expr)
    --> 337         _check_for_locals(expr, level, parser)
        338 
        339         # get our (possibly passed-in) scope


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/core/computation/eval.py in _check_for_locals(expr, stack_level, parser)
        157 
        158     if at_top_of_stack or not_pandas_parser:
    --> 159         for toknum, tokval in tokenize_string(expr):
        160             if toknum == tokenize.OP and tokval == "@":
        161                 raise SyntaxError(msg)


    ~/opt/miniconda3/envs/neocov/lib/python3.10/site-packages/pandas/core/computation/parsing.py in tokenize_string(source)
        183     # Loop over all tokens till a backtick (`) is found.
        184     # Then, take all tokens till the next backtick to form a backtick quoted string
    --> 185     for toknum, tokval, start, _, _ in token_generator:
        186         if tokval == "`":
        187             try:


    ~/opt/miniconda3/envs/neocov/lib/python3.10/tokenize.py in _tokenize(readline, encoding)
        533                 token, initial = line[start:end], line[start]
        534 
    --> 535                 if (initial in numchars or                 # ordinary number
        536                     (initial == '.' and token != '.' and token != '...')):
        537                     yield TokenInfo(NUMBER, token, spos, epos, line)


    KeyboardInterrupt: 


```python
docs_clean
```




    Series([], Name: ['if', 'this', 'is', 'a', 'dank', 'meme', 'upvote', 'this', 'comment', 'if', 'this', 'is', 'not', 'a', 'dank', 'meme', 'downvote', 'this', 'comment', 'if', 'this', 'post', 'breaks', 'the', 'rules', 'report', 'it', 'and', 'downvote', 'this', 'comment', 'thank', 'you', 'for', 'helping', 'us', 'in', 'keeping', 'r', 'dankmemes', 'dank', 'hit', 'us', 'up', 'https', 'www', 'reddit', 'com', 'message', 'compose', 'to', 'r', 'dankmemes', 'if', 'you', 'have', 'any', 'questions', 'i', 'm', 'a', 'bot'], dtype: float64)



##### from `parquet`

```python
%%time
docs_clean = pd.read_parquet(f'~/promo/socemb/data/docs_clean/{YEAR}.parquet')
```

    CPU times: user 18.8 s, sys: 6.62 s, total: 25.5 s
    Wall time: 55.2 s


```python
%%time
docs_clean['body'] = docs_clean['body'].apply(lambda x: x.tolist())
```

```python
docs_clean = docs_clean['body']
```

```python
docs_clean
```

## Train models

### Create corpus

```python
corpus = Corpus(docs_clean)
```

### Train model

```python
from gensim.models import Word2Vec
```

```python
%%time
model = train_emb(corpus)
```

    CPU times: user 54min 54s, sys: 3min 16s, total: 58min 11s
    Wall time: 24min 2s


```python
len(model.wv.key_to_index)
```




    244740



```python
len(model.wv.key_to_index)
```




    244740



### Save model

```python
model.save(f'../out/models/{YEAR}.model')
```

### Load models

```python
model_2019 = gensim.models.Word2Vec.load('out/models/2019.model')
```

```python
model_2020 = gensim.models.Word2Vec.load('out/models/2020.model')
```
