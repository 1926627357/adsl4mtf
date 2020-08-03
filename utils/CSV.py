import numpy as np
import pandas as pd

class CSV:
    def __init__(self, path, columns, values):
        self.path = path
        self.columns = columns
        self.values = values
    def dump(self):
        self.values = np.array(self.values).T
        self.values = self.values.tolist()
        dataframe = pd.DataFrame(columns=self.columns,data=self.values)
        dataframe.to_csv(path_or_buf=self.path,index=False,sep=',')


if __name__ == "__main__":
    instance = CSV(path="./test.csv",columns=['epoch','loss'],values=[[1,2],[3,4]])
    instance.dump()