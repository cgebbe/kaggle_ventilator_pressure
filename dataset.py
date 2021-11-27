import pandas as pd
import numpy as np


class Dataset:
    def __init__(self, path):
        self.df = pd.read_csv(path)
        self.id_by_idx = {
            idx: breathid for idx, breathid in enumerate(self.df["breath_id"].unique())
        }

    def __len__(self):
        return len(self.id_by_idx)

    def __getitem__(self, idx):
        breathid = self.id_by_idx[idx]
        mask = self.df["breath_id"] == breathid
        tmp = self.df.loc[mask, :]
        return {c: tmp[c].values for c in tmp.columns}


if __name__ == "__main__":
    ds_train = Dataset("data/train_subset.csv")
    print(ds_train[3])
