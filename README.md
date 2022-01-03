# NeoCov
> Semantic change and social semantic variation of Covid-related English neologisms on Reddit.


```python
%load_ext autoreload
%autoreload 2
```

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload


# Read data

```python
COMMENTS_DIR = '../data/comments/by_date/'
```

```python
YEAR = 2019
```

```python
get_comments_paths_year(COMMENTS_DIR, '2019')
```




    [PosixPath('../data/comments/by_date/2019-05-07_21:11:36___2019-05-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-08-07_21:12:15___2019-08-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-07-14_21:06:51___2019-07-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-07-01_21:59:59___2019-07-01_21:19:55.csv'),
     PosixPath('../data/comments/by_date/2019-05-14_21:15:37___2019-05-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-06-07_21:17:11___2019-06-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-02-01_22:59:59___2019-02-01_22:02:38.csv'),
     PosixPath('../data/comments/by_date/2019-02-07_22:06:26___2019-02-07_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-06-01_21:59:59___2019-06-01_21:09:37.csv'),
     PosixPath('../data/comments/by_date/2019-11-07_22:24:23___2019-11-07_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-03-14_22:06:05___2019-03-14_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-10-07_21:17:33___2019-10-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-02-14_22:04:57___2019-02-14_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-12-07_22:29:19___2019-12-07_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-08-01_21:59:59___2019-08-01_21:13:12.csv'),
     PosixPath('../data/comments/by_date/2019-03-01_22:59:59___2019-03-01_22:04:58.csv'),
     PosixPath('../data/comments/by_date/2019-04-19_21:03:20___2019-04-19_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-06-14_21:02:30___2019-06-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-01-19_22:12:29___2019-01-19_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-11-14_22:11:50___2019-11-14_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-01-01_22:59:59___2019-01-01_21:59:07.csv'),
     PosixPath('../data/comments/by_date/2019-12-19_22:09:07___2019-12-19_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-12-01_22:59:59___2019-12-01_22:36:30.csv'),
     PosixPath('../data/comments/by_date/2019-04-01_21:59:59___2019-04-01_21:05:49.csv'),
     PosixPath('../data/comments/by_date/2019-11-01_22:59:59___2019-11-01_22:06:28.csv'),
     PosixPath('../data/comments/by_date/2019-10-01_21:59:59___2019-10-01_21:14:05.csv'),
     PosixPath('../data/comments/by_date/2019-09-19_21:12:20___2019-09-19_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-01-07_22:14:11___2019-01-07_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-04-14_21:01:38___2019-04-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-07-19_21:10:09___2019-07-19_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-05-01_21:59:59___2019-05-01_21:10:03.csv'),
     PosixPath('../data/comments/by_date/2019-09-14_21:17:45___2019-09-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-03-19_22:08:57___2019-03-19_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-09-01_21:59:59___2019-09-01_21:03:54.csv'),
     PosixPath('../data/comments/by_date/2019-01-14_22:04:48___2019-01-14_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-09-07_21:09:47___2019-09-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-10-14_21:33:09___2019-10-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-08-19_21:10:07___2019-08-19_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-10-19_21:04:06___2019-10-19_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-12-14_22:34:00___2019-12-14_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-03-07_22:06:55___2019-03-07_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-02-19_22:06:50___2019-02-19_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-08-14_21:13:49___2019-08-14_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-05-19_21:16:29___2019-05-19_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-07-07_21:07:10___2019-07-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-04-07_21:10:44___2019-04-07_21:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-11-19_22:15:40___2019-11-19_22:59:59.csv'),
     PosixPath('../data/comments/by_date/2019-06-19_21:09:12___2019-06-19_21:59:59.csv')]


