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

# Read data

### Get file paths

```python
COMMENTS_DIR = '../data/comments/by_date/'
```

```python
YEAR = 2020
```

```python
comment_paths_year = get_comments_paths_year(COMMENTS_DIR, YEAR)
```

### Read comments

```python
%%time
comments = read_comm_csvs(comment_paths_year)
```

```python
comments.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 9599970 entries, 0 to 9599969
    Data columns (total 5 columns):
     #   Column       Dtype         
    ---  ------       -----         
     0   author       string        
     1   body         string        
     2   created_utc  datetime64[ns]
     3   id           string        
     4   subreddit    string        
    dtypes: datetime64[ns](1), string(4)
    memory usage: 366.2 MB


```python
comments.value_counts('subreddit')
```




    subreddit
    AskReddit              341545
    politics               136751
    memes                  129039
    WatchUFC248Stream       75756
    wallstreetbets          75698
                            ...  
    TimTeemoSubmissions         1
    TimberAndStone              1
    Timberland                  1
    TimeTrap                    1
    zztails                     1
    Length: 93128, dtype: int64



### Pre-process comments

#### run preprocessing

```python
%%time
docs_clean = clean_docs(comments['body'])
```

    CPU times: user 2min 17s, sys: 7min 11s, total: 9min 29s
    Wall time: 59min


```python
docs_clean
```




    0          [oh, okay, thank, you, so, much, for, the, rep...
    1          [es, tan, deprimente, ver, cuando, esta, clase...
    4          [am, i, the, only, person, who, thinks, this, ...
    5          [sorry, this, happened, in, hamilton, ontario,...
    7          [that, was, awesome, man, i, want, to, be, you...
                                     ...                        
    9599956    [the, third, downside, is, that, if, you, upda...
    9599959    [what, do, you, mean, britain, stood, alone, i...
    9599962    [flat, earth, started, as, a, joke, i, imagine...
    9599963    [the, only, canonical, way, for, them, to, be,...
    9599969    [yeah, its, weird, i, think, jamal, has, passe...
    Name: body, Length: 5385571, dtype: object



#### save to disk

```python
%%time
docs_clean.to_csv(f'../data/docs_clean/{YEAR}.csv', index=False)
```

    CPU times: user 51.3 s, sys: 36.7 s, total: 1min 28s
    Wall time: 3min 55s


#### load from disk

```python
import pandas as pd
```

```python
%%time
docs_clean = pd.read_csv(f'../data/docs_clean/{YEAR}.csv', index_col=0, header=None)
```

    CPU times: user 20.2 s, sys: 2.47 s, total: 22.6 s
    Wall time: 28.6 s

