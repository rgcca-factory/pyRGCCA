from pyRGCCA.rgcca import SGCCA

import pandas as pd
import numpy as np

data = pd.read_csv('./examples/data/russett.csv', index_col=0)
blocks = [data.iloc[:, :3].to_numpy(),
        data.iloc[:, 3:5].to_numpy(),   
        data.iloc[:, 5:].to_numpy()]

sgcca = SGCCA(
        n_components=2,
        connection=np.ones((3,3))-np.eye(3), 
        sparsity=[0.8]*3, 
        init="svd", 
        scheme="factorial"        
        )

sgcca.fit(blocks)