import pandas as pd
import numpy as np
import torch
import glob
import os

def get_stitched_data(path):
    files = sorted(glob.glob(path+'/*/*.pt'))
    main = torch.load(files[0])

    for file in files[1:]:
        ckpt = torch.load(file)['M']
        main['M'] = torch.cat([main['M'], ckpt], dim=0)

    return main
