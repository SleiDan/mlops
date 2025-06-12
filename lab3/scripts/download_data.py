import pandas as pd
from sklearn.datasets import load_iris

df = pd.DataFrame(load_iris(as_frame=True).frame)
df.to_csv("data/iris.csv", index=False)
