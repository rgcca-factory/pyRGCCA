from pyRGCCA.rgcca import GlobalRGCCA

import pandas as pd
import numpy as np

data = pd.read_csv('./examples/data/russett.csv', index_col=0)
blocks = [data.iloc[:, :3].to_numpy(),
        data.iloc[:, 3:5].to_numpy(),   
        data.iloc[:, 5:].to_numpy()]

grgcca = GlobalRGCCA(
        n_components=2, 
        connection=np.ones((3,3))-np.eye(3), 
        tau=[1]*3,
        init="random", 
        scheme="factorial", 
        tol=1e-8
        )

grgcca.fit(blocks)

print(grgcca.loadings_)