from pyRGCCA.rgcca import RGCCA

import pandas as pd
import numpy as np

data = pd.read_csv('./examples/data/russett.csv', index_col=0)
blocks = [data.iloc[:, :3].to_numpy(),
        data.iloc[:, 3:5].to_numpy(),   
        data.iloc[:, 5:].to_numpy()]

rgcca = RGCCA(
        connection=np.ones((3,3))-np.eye(3), 
        tau=[1]*3, 
        init="svd", 
        scheme="factorial", 
        n_components=2
        )

rgcca.fit(blocks)

print(rgcca.loadings_)