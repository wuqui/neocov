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
YEAR = 2019
```

```python
comment_paths_year = get_comments_paths_year(COMMENTS_DIR, '2019')
```

### Read comments

```python
comments = read_comm_csvs(comment_paths_year)
```

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



### Pre-process comments

#### run preprocessing

```python
%%time
docs_clean = clean_docs(comments['body'])
```

```python
docs_clean
```




    1          [if, this, is, a, dank, meme, upvote, this, co...
    2          [just, threaten, them, that, you, ll, call, th...
    4          [honestly, do, you, really, wanna, go, through...
    5          [i, actually, think, they, wouldn, t, have, pu...
    6          [as, a, girl, on, the, sub, i, laughed, at, th...
                                     ...                        
    9599968    [i, think, they, would, interpret, residential...
    9599971    [16, is, a, very, young, age, they, do, a, lot...
    9599972    [i, would, ve, downvoted, but, since, you, adm...
    9599974    [guy, who, made, the, crossbuck, had, one, job...
    9599978    [this, is, barely, even, half, an, inch, by, t...
    Name: body, Length: 5308119, dtype: object



#### save to disk

```python
%%time
docs_clean.to_csv(f'../data/docs_clean/{YEAR}.csv', index=False)
```

    CPU times: user 49.2 s, sys: 22 s, total: 1min 11s
    Wall time: 2min 19s


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

