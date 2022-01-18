# AUTOGENERATED! DO NOT EDIT! File to edit: 03_communities.ipynb (unless otherwise specified).

__all__ = ['get_subr_counts', 'plot_subr_counts']

# Cell
import altair as alt

# Cell
def get_subr_counts(comments):
    subr_counts = comments\
        .groupby('subreddit')\
        .agg(comments_num = ('subreddit', 'count'))\
        .sort_values('comments_num', ascending=False)
    return subr_counts

# Cell
def plot_subr_counts(subr_counts, k=20):
    chart = subr_counts\
        .reset_index()\
        .iloc[:k]\
        .pipe(alt.Chart)\
            .mark_bar()\
            .encode(
                x=alt.X('comments_num:Q'),
                y=alt.Y('subreddit:N', sort='-x')
            )
    return chart