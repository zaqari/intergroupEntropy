import pandas as pd
import numpy as np
import regex as re

pattern_example = "([Ww]oman|[Ww]omen|[Ff]eminis)"

def find_pattern(df, col, pattern):
    search = re.compile(pattern)
    df['pattern_found'] = [bool(re.findall(search, df[col].loc[j])) for j in df.index]
    return df

