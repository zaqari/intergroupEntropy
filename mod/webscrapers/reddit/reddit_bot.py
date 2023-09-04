import pandas as pd
import praw
from praw.models import MoreComments
import regex as re
import numpy as np

def author_name(x):
    if x == None:
        return None
    else:
        return x.name


###########################################################################################
#### Reddit set-up
###########################################################################################
bot = praw.Reddit(
    client_id="s19hRA227GIURi6mEsuREQ",
    client_secret="hFMidkmAnAXkHHvOn2iSQyYTki3RpQ",
    user_agent="PhisherFinderDestroyer",
    username="PhisherAvenger",
    password="7Mojgani7&"
)

def bind_quote_text(text):
    quote_start = [i.span()[0] for i in list(re.finditer('\n>', text))]
    if text[0] == '>':
        quote_start = [0] + quote_start

    if len(quote_start) > 0:
        quote_start = np.array(quote_start)
        quote_ends = np.array([i.span()[0] for i in list(re.finditer('\n', text))])
        splits = [(None, quote_start[0])]+[
            (start, quote_ends[quote_ends > start][0])
            if (len(quote_ends[quote_ends > start]) > 0)
            else (start, None)
            for start in quote_start
        ]

        splits += [(splits[-1][-1], None)]
        return '<QUOTE>'.join([text[split[0]: split[1]] for split in splits])

    else: return text

def _flatten_comment_tree(submission):
    comments = submission.comments.list()
    for comment in comments:
        if isinstance(comment,MoreComments):
            comments+=comment.comments()
    return [[comment.subreddit.display_name, submission.id, submission.created_utc, comment.id, comment.ups, comment.created_utc, author_name(comment.author), comment.body.lower()] for comment in comments if hasattr(comment, 'body')]

def flatten_comment_tree(submission):
    comments = submission.comments.list()
    for comment in comments:
        if isinstance(comment,MoreComments):
            pass
        else:
            comments+=comment.replies.list()

    comments = [[comment.subreddit.display_name, submission.id, submission.created_utc, comment.id, comment.ups, comment.created_utc, author_name(comment.author), bind_quote_text(comment.body.lower())] for comment in comments if hasattr(comment, 'body')]
    comments = pd.DataFrame(np.array(comments), columns=['subreddit', 'sub_id', 'sub_create_at', 'comment_id', 'comment_ups', 'comment_created_at', 'user', 'body']).drop_duplicates()
    comments.index = range(len(comments))

    return comments


def get_submissions(subreddit,title_terms,min_date,max_date):
    sub = bot.subreddit(subreddit)
    sids = []
    for term in title_terms:
        sids += [
            submission.id for submission in sub.search(term)
            if (submission.created_utc > min_date)
               and (submission.created_utc < max_date)
        ]
    return list(set(sids))