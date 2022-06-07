from webscrapers.reddit.scraper import *

search_terms = ['woman', 'women', 'girls']
get_comments = True
subreddits = ['feminism', 'mensrights', 'menslib']

if get_comments:
    dfs = {sub:get_top_posts_monthly_comments(sub) for sub in subreddits}
else:
    dfs = {sub: get_top_posts_monthly(sub) for sub in subreddits}
dfs = pd.concat(list(dfs.values()), ignore_index=True)

keep_indeces = [i for i in dfs.index if sum([w in dfs['post_title'].loc[i].lower() for w in search_terms]) > 0]
dfs=dfs.loc[keep_indeces].copy()

output_file = "/Volumes/V'GER/comp_ling/DataScideProjects/convergenceEntropy/data/redditany/three_groups/woman_post_title/reddit_{}.csv".format('-'.join(subreddits))
dfs.to_csv(output_file, index=False, encoding='utf-8')
