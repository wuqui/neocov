# AUTOGENERATED! DO NOT EDIT! File to edit: 00_read_data.ipynb (unless otherwise specified).

__all__ = ['get_comments_paths_year', 'get_comments_paths_subr', 'read_comm_csv', 'read_comm_csvs']

# Cell
from pathlib import Path
import pandas as pd

# Cell
def get_comments_paths_year(COMMENTS_DIR, YEAR):
    comments_dir_path = Path(COMMENTS_DIR)
    comments_paths = list(comments_dir_path.glob(f'{YEAR}*.csv'))
    return comments_paths

# Cell
def get_comments_paths_subr(COMMENTS_DIR_SUBR, SUBR):
	comments_subr_dir_path = Path(COMMENTS_DIR_SUBR)
	comments_subr_paths = list(comments_subr_dir_path.glob(f'{SUBR}*.csv'))
	return comments_subr_paths

# Cell
def read_comm_csv(fpath):
    try:
        # removed because new method for writing retrieved data out already does date conversion beforehand
        # date_parser = lambda x: pd.to_datetime(x, unit='s', errors='coerce')
        comments = pd.read_csv(
            fpath,
            usecols=['id', 'created_utc', 'author', 'subreddit', 'body'],
            dtype={
                'id': 'string',
                # 'created_utc': int, s. above
                'author': 'string',
                'subreddit': 'string',
                'body': 'string'
            },
            parse_dates=['created_utc'],
            # date_parser=date_parser,
            low_memory=False,
            lineterminator='\n'
        )
        comments_clean = comments\
            .dropna()\
            .drop_duplicates(subset='id')
        return comments_clean
    except FileNotFoundError:
        print(f'{fpath} not found on disk')
    except pd.errors.EmptyDataError:
        print(f'{fpath} is empty')

# Cell
def read_comm_csvs(fpaths: list):
    comments_lst = []
    for fpath in fpaths:
        comments = read_comm_csv(fpath)
        comments_lst.append(comments)
    comments_concat = pd.concat(
        comments_lst,
        axis=0,
        ignore_index=True
    )
    return comments_concat