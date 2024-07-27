from typing import Optional, Dict
import os
import pandas as pd
import numpy as np
from mindsdb.utilities import log
from mindsdb.integrations.libs.base import BaseMLEngine
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle

logger = log.getLogger(__name__)


class NgxClusteringHandler(BaseMLEngine):
    name = "ngxclustering"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(
        self,
        target: str,
        df: Optional[pd.DataFrame] = None,
        args: Optional[Dict] = None,
    ) -> None:
        """Create and train model on given data"""
        # parse args
        if "using" not in args:
            raise Exception(
                "PyCaret engine requires a USING clause! Refer to its documentation for more details."
            )
        using = args["using"]
        if df is None:
            raise Exception("PyCaret engine requires a some data to initialize!")

        using["orderby_cols"] = using["orderby_cols"] if "orderby_cols" in using else []
        using["omit_cols"] = using["omit_cols"] if "omit_cols" in using else []
        using["cluster_descriptions"] = (
            using["cluster_descriptions"] if "cluster_descriptions" in using else []
        )

        for c in using["omit_cols"]:
            df[c] = df[c].astype("string")

        X_cols = []
        for c in df.columns:
            if c not in using["omit_cols"] + [target]:
                X_cols.append(c)

        using["n_clusters"] = args.get("n_clusters", 4)
        using["random_state"] = args.get("random_state", 1)
        using["target"] = target

        model_file_path = os.path.join(
            self.model_storage.fileStorage.folder_path, "model"
        )

        if df.shape[0] >= 4:
            df_log = np.log1p(df[X_cols].astype(np.float32)).copy()
            scaler = MinMaxScaler()
            df_norm = scaler.fit_transform(df_log)

            kmeans = KMeans(n_clusters=using["n_clusters"], random_state=1)
            kmeans.fit(df_norm)

            with open(model_file_path, "wb") as f:
                pickle.dump(kmeans, f)

            self.model_storage.json_set(
                "saved_args", {**using, "model_path": model_file_path}
            )

    def predict(
        self, df: Optional[pd.DataFrame] = None, args: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Predict on the given data"""
        # load model
        saved_args = self.model_storage.json_get("saved_args")
        with open(saved_args["model_path"], "rb") as f:
            model = pickle.load(f)

        for c in saved_args["omit_cols"]:
            df[c] = df[c].astype("string")

        df[saved_args["target"]] = model.labels_

        X_cols = []
        for c in df.columns:
            if c not in saved_args["omit_cols"]:
                X_cols.append(c)

        table_params = (
            df[X_cols].groupby(by=[saved_args["target"]]).agg("mean").reset_index()
        )
        table_params[saved_args["omit_cols"]] = df[saved_args["omit_cols"]]

        table_params = table_params[
            [saved_args["target"]] + saved_args["orderby_cols"]
        ].sort_values(by=saved_args["orderby_cols"], ascending=False)

        df_cat = pd.DataFrame(
            {
                "cluster_descriptions": saved_args["cluster_descriptions"],
                saved_args["target"]: table_params[saved_args["target"]],
            }
        )

        df = df.join(df_cat, on=saved_args["target"], how="inner", rsuffix="_alt").drop(
            saved_args["target"] + "_alt", axis=1
        )

        return df

    def describe(self, attribute=None):
        model_args = self.model_storage.json_get("model_args")

        if attribute == "model":
            return pd.DataFrame(
                {k: [model_args[k]] for k in ["model_name", "frequency", "hierarchy"]}
            )

        elif attribute == "features":
            return pd.DataFrame(
                {
                    "ds": [model_args["order_by"]],
                    "y": model_args["target"],
                    "unique_id": [model_args["group_by"]],
                    "exog_vars": [model_args["exog_vars"]],
                }
            )

        elif attribute == "info":
            outputs = model_args["target"]
            inputs = [
                model_args["target"],
                model_args["order_by"],
                model_args["group_by"],
            ] + model_args["exog_vars"]
            accuracies = [
                (model, acc) for model, acc in model_args.get("accuracies", {}).items()
            ]
            return pd.DataFrame(
                {"accuracies": [accuracies], "outputs": outputs, "inputs": [inputs]}
            )

        else:
            tables = ["info", "features", "model"]
            return pd.DataFrame(tables, columns=["tables"])
