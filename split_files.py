import pandas as pd
import sklearn.model_selection

if __name__ == "__main__":
    train_size = 5_000
    valid_size = 2_000

    train_org = pd.read_csv("data/org/train.csv", index_col="id")
    unique_breath_ids = train_org["breath_id"].unique()
    train_breaths, valid_breaths = sklearn.model_selection.train_test_split(
        unique_breath_ids, train_size=train_size, test_size=valid_size
    )
    assert len(train_breaths) == train_size
    assert len(valid_breaths) == valid_size

    train_org.loc[train_org["breath_id"].isin(train_breaths), :].to_csv(
        f"data/train_{train_size}.csv"
    )
    train_org.loc[train_org["breath_id"].isin(valid_breaths), :].to_csv(
        f"data/valid_{valid_size}.csv"
    )

    # reopen
    tmp = pd.read_csv(f"data/valid_{valid_size}.csv")
    assert tmp["breath_id"].nunique() == valid_size
