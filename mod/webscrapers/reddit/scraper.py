import praw
import pandas as pd
import numpy as np

bot = praw.Reddit(
    client_id="",
    client_secret="",
    user_agent="",
    username="",
    password=""
)

def get_top_posts_monthly(subreddit):
    sr = bot.subreddit(subreddit)
    data = [[subreddit, post.id,  post.title, post.selftext] for post in sr.top("month")]
    data = np.array(data, dtype='object')
    return pd.DataFrame(data, columns=['subreddit_name', 'post_id',  'post_title', 'text'])

deep_errors = []

def get_top_posts_monthly_comments(subreddit, search_terms, deep_error_list=deep_errors):
    sr = bot.subreddit(subreddit).top("month")
    data = []
    for post in sr:
        if sum([w.lower() in post.title.lower() for w in search_terms]) > 0:
            post_data = [subreddit, post.id, post.title, post.ups, post.created_utc]
            for comment in post.comments:
                try:
                    data += [post_data+[comment.ups, comment.created_utc, comment.body]]
                except AttributeError:
                    for c in comment.comments():
                        try:
                            data += [post_data+[c.ups, c.created_utc, c.body]]
                        except AttributeError:
                            deep_error_list += [[subreddit, post.id, post.title, post]]
    return pd.DataFrame(np.array(data), columns=['subreddit_name',
                                                 'post_id', 'post_title', 'post_ups', 'post_time',
                                                 'comment_ups', 'comment_time', 'comment'])

def retroactive_comments_created():
    0