# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

#import os
#os.environ['NIXTLA_ID_AS_COL'] = '1'

from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import tempfile
from mindsdb.integrations.libs.base import BaseMLEngine
# from mindsdb.integrations.utilities.time_series_utils import (
#     get_model_accuracy_dict,
# )
from mindsdb.utilities import log
from neuralforecast import NeuralForecast
from neuralforecast.models import (
    LSTM,
    GRU,
    RNN,
    DilatedRNN,
    DeepAR,
    TCN,
    TimesNet,
    MLP,
    NBEATS,
    NBEATSx,
    NHITS,
    TFT,
    VanillaTransformer,
    Informer,
    Autoformer,
    FEDformer,
    PatchTST,
    StemGNN,
)
from neuralforecast.auto import (
    AutoLSTM,
    AutoGRU,
    AutoRNN,
    AutoDilatedRNN,
    AutoDeepAR,
    AutoTCN,
    AutoTimesNet,
    AutoMLP,
    AutoNBEATS,
    AutoNBEATSx,
    AutoNHITS,
    AutoTFT,
    AutoVanillaTransformer,
    AutoInformer,
    AutoAutoformer,
    AutoFEDformer,
    AutoPatchTST,
    AutoStemGNN,
)
from ray.tune.search.hyperopt import HyperOptSearch
from prophet.make_holidays import make_holidays_df

logger = log.getLogger(__name__)


def transform_to_nixtla_df(df, settings_dict, exog_vars=[]):
    """Transform dataframes into the specific format required by StatsForecast.

    Nixtla packages require dataframes to have the following columns:
        unique_id -> the grouping column. If multiple groups are specified then
        we join them into one name using a / char.
        ds -> the date series
        y -> the target variable for prediction

    You can optionally include exogenous regressors after these three columns, but
    they must be numeric.
    """
    nixtla_df = df.copy()

    # Transform group columns into single unique_id column
    if len(settings_dict["group_by"]) > 1:
        for col in settings_dict["group_by"]:
            nixtla_df[col] = nixtla_df[col].astype(str)
        nixtla_df["unique_id"] = nixtla_df[settings_dict["group_by"]].agg(
            "/".join, axis=1
        )
        group_col = "ignore this"
    else:
        group_col = settings_dict["group_by"][0]

    # Rename columns to statsforecast names
    nixtla_df = nixtla_df.rename(
        {
            settings_dict["target"]: "y",
            settings_dict["order_by"]: "ds",
            group_col: "unique_id",
        },
        axis=1,
    )

    if "unique_id" not in nixtla_df.columns:
        # add to dataframe as it is expected by statsforecast
        nixtla_df["unique_id"] = "1"

    nixtla_df["ds"] = pd.to_datetime(nixtla_df["ds"])

    for prop in settings_dict["ds_props"]:
        if prop == "year":
            nixtla_df.loc[:, "year"] = nixtla_df["ds"].dt.year
        elif prop == "month":
            nixtla_df.loc[:, "month"] = nixtla_df["ds"].dt.month
        elif prop == "day":
            nixtla_df.loc[:, "day"] = nixtla_df["ds"].dt.day
        elif prop == "dayofweek":
            nixtla_df.loc[:, "dayofweek"] = nixtla_df["ds"].dt.dayofweek
        elif prop == "dayofyear":
            nixtla_df.loc[:, "dayofyear"] = nixtla_df["ds"].dt.dayofyear
        elif prop == "quarter":
            nixtla_df.loc[:, "quarter"] = nixtla_df["ds"].dt.quarter
        elif prop == "weekofyear":
            nixtla_df.loc[:, "weekofyear"] = nixtla_df["ds"].dt.isocalendar().week
        elif prop == "holiday":
            holidays = make_holidays_df(
                year_list=nixtla_df["ds"].dt.year.unique(),
                country=settings_dict["ds_holiday_country"],
            )
            holidays["holiday"] = 1.0
            holidays = holidays.convert_dtypes({"holiday": np.float16})

            nixtla_df = pd.merge(nixtla_df, holidays, how="left", on=["ds"])
            nixtla_df["holiday"] = nixtla_df["holiday"].fillna(0.0)

    # lags_cols = []
    # for lag in range(1, settings_dict["lags"]):
    #     nixtla_df.loc[:, f"lag_[{lag}]"] = nixtla_df.groupby(["unique_id"])["y"].shift(
    #         lag
    #     )
    #     lags_cols.append(f"lag_[{lag}]")

    # nixtla_df[lags_cols] = nixtla_df[lags_cols].fillna(0.0)

    columns_to_keep = (
        ["unique_id", "ds", "y"] + exog_vars + settings_dict["ds_props"] # + lags_cols
    )

    return nixtla_df[columns_to_keep]


def get_results_from_nixtla_df(nixtla_df, model_args):
    """Transform dataframes generated by StatsForecast back to their original format.
    This will return the dataframe to the original format supplied by the MindsDB query.
    """
    return_df = nixtla_df.reset_index(
        drop=True if "unique_id" in nixtla_df.columns else False
    )
    if len(model_args["group_by"]) > 0:     
        if len(model_args["group_by"]) > 1:
            for i, group in enumerate(model_args["group_by"]):
                return_df[group] = return_df["unique_id"].apply(
                    lambda x: x.split("/")[i]
                )
        else:
            group_by_col = model_args["group_by"][0]
            return_df[group_by_col] = return_df["unique_id"]
    
    res = return_df.drop(["unique_id"], axis=1).rename(
        {"ds": model_args["order_by"]}, axis=1
    )

    return res


def get_model_accuracy_dict(nixtla_results_df, metric=r2_score):
    """Calculates accuracy for each model in the nixtla results df."""
    accuracy_dict = {}
    for column in nixtla_results_df.columns:
        if column in ["unique_id", "ds", "y", "cutoff"]:
            continue
        model_error = metric(nixtla_results_df["y"], nixtla_results_df[column])
        accuracy_dict[column] = model_error
    return accuracy_dict


class NgxForecastHandler(BaseMLEngine):
    """Integration with the Nixtla NeuralForecast library for
    time series forecasting with neural networks.
    """

    name = "ngxforecast"

    def __init__(self, model_storage, engine_storage, **kwargs):
        super().__init__(model_storage, engine_storage, **kwargs)
        # self.model_cache = {}

    def create(self, target, df, args={}):
        """Create the NeuralForecast Handler.

        Requires specifying the target column to predict and time series arguments for
        prediction horizon, time column (order by) and grouping column(s).

        Saves model params to desk, which are called later in the predict() method.
        """
        time_settings = args["timeseries_settings"]
        using_args = args["using"]
        assert time_settings[
            "is_timeseries"
        ], "Specify time series settings in your query"

        model_args = {}
        model_args["target"] = target
        model_args["horizon"] = time_settings["horizon"]
        model_args["order_by"] = time_settings["order_by"]
        model_args["group_by"] = time_settings["group_by"]
        model_args["frequency"] = (
            using_args["frequency"]
            if "frequency" in using_args
            else "D"
        )
        model_args["scaler_type"] = using_args.get("scaler_type", None)
        model_args["local_scaler_type"] = using_args.get(
            "local_scaler_type", None
        )
        model_args["exog_vars"] = (
            using_args["exogenous_vars"] if "exogenous_vars" in using_args else []
        )
        model_args["max_steps"] = using_args.get("max_steps", 500)
        model_args["val_check_steps"] = using_args.get("val_check_steps", None)
        model_args["n_auto_trials"] = using_args.get("n_auto_trials", None)
        model_args["model_folder"] = tempfile.mkdtemp()

        model_args["encoder_hidden_size"] = using_args.get("encoder_hidden_size", None)
        model_args["decoder_hidden_size"] = using_args.get("decoder_hidden_size", None)

        model_args["encoder_n_layers"] = using_args.get("encoder_n_layers", None)
        model_args["decoder_layers"] = using_args.get("decoder_layers", None)

        model_args["batch_size"] = using_args.get("batch_size", 32)
        model_args["context_size"] = using_args.get("context_size", None)

        # model_args["lags"] = using_args.get("lags", 0)
        model_args["ds_props"] = (
            using_args["ds_props"] if "ds_props" in using_args else []
        )
        model_args["ds_holiday_country"] = using_args.get("ds_holiday_country", "CO")

        model_args["model_type"] = using_args.get("model_type", "lstm")

        model_args["n_series"] = using_args.get("n_series", None)

        # Deal with hierarchy
        # model_args["hierarchy"] = using_args["hierarchy"] if "hierarchy" in using_args else False
        # if model_args["hierarchy"] and HierarchicalReconciliation is not None:
        #     training_df, hier_df, hier_dict = get_hierarchy_from_df(df, model_args)
        #     self.model_storage.file_set("hier_dict", dill.dumps(hier_dict))
        #     self.model_storage.file_set("hier_df", dill.dumps(hier_df))
        #     self.model_storage.file_set("training_df", dill.dumps(training_df))
        # else:
        training_df = transform_to_nixtla_df(df, model_args, model_args["exog_vars"])

        training_df = training_df.astype({"y": np.float32})

        model = None

        # Model	Structure	Sampling	Point Forecast	Probabilistic Forecast	Exogenous features	Auto Model
        # LSTM	RNN	recurrent	✅	✅	✅	✅
        # GRU	RNN	recurrent	✅	✅	✅	✅
        # RNN	RNN	recurrent	✅	✅	✅	✅
        # DilatedRNN	RNN	recurrent	✅	✅	✅	✅
        # DeepAR	RNN	recurrent		✅	✅	✅
        # TCN	CNN	recurrent	✅	✅	✅	✅
        # TimesNet	CNN	windows	✅	✅		✅
        # DLinear	Linear	windows	✅	✅		✅
        # MLP	MLP	windows	✅	✅	✅	✅
        # NBEATS	MLP	windows	✅	✅		✅
        # NBEATSx	MLP	windows	✅	✅	✅	✅
        # NHITS	MLP	windows	✅	✅	✅	✅
        # TFT	Transformer	windows	✅	✅	✅	✅
        # Transformer	Transformer	windows	✅	✅	✅	✅
        # Informer	Transformer	windows	✅	✅	✅	✅
        # Autoformer	Transformer	windows	✅	✅	✅	✅
        # FEDFormer	Transformer	windows	✅	✅	✅	✅
        # PatchTST	Transformer	windows	✅	✅		✅
        # StemGNN	GNN	multivariate	✅			✅

        # Train model
        if model_args["n_auto_trials"]:
            conf = {
                "h": time_settings["horizon"],
                "gpus": 0,
                "num_samples": model_args["n_auto_trials"],
                "search_alg": HyperOptSearch(),
            }
            if model_args["model_type"].lower() == "lstm":
                model = AutoLSTM(**conf)
            elif model_args["model_type"].lower() == "gru":
                model = AutoGRU(**conf)
            elif model_args["model_type"].lower() == "rnn":
                model = AutoRNN(**conf)
            elif model_args["model_type"].lower() == "dilatedrnn":
                model = AutoDilatedRNN(**conf)
            elif model_args["model_type"].lower() == "deepar":
                model = AutoDeepAR(**conf)
            elif model_args["model_type"].lower() == "tcn":
                model = AutoTCN(**conf)
            elif model_args["model_type"].lower() == "timesnet":
                model = AutoTimesNet(**conf)
            elif model_args["model_type"].lower() == "mlp":
                model = AutoMLP(**conf)
            elif model_args["model_type"].lower() == "nbeats":
                model = AutoNBEATS(**conf)
            elif model_args["model_type"].lower() == "nbeatsx":
                model = AutoNBEATSx(**conf)
            elif model_args["model_type"].lower() == "nhits":
                model = AutoNHITS(**conf)
            elif model_args["model_type"].lower() == "tft":
                model = AutoTFT(**conf)
            elif model_args["model_type"].lower() == "vanillatransformer":
                model = AutoVanillaTransformer(**conf)
            elif model_args["model_type"].lower() == "informer":
                model = AutoInformer(**conf)
            elif model_args["model_type"].lower() == "autoformer":
                model = AutoAutoformer(**conf)
            elif model_args["model_type"].lower() == "fedformer":
                model = AutoFEDformer(**conf)
            elif model_args["model_type"].lower() == "patchtst":
                model = AutoPatchTST(**conf)
            elif model_args["model_type"].lower() == "stemgnn":
                model = AutoStemGNN(**conf)
        else:
            conf = {
                "h": time_settings["horizon"],
                "input_size": time_settings["window"], 
                "scaler_type": model_args["scaler_type"],
                "encoder_hidden_size": model_args["encoder_hidden_size"],
                "decoder_hidden_size": model_args["decoder_hidden_size"],
                "encoder_n_layers": model_args["encoder_n_layers"],
                "decoder_layers": model_args["decoder_layers"],
                "batch_size": model_args["batch_size"],
                "context_size": model_args["context_size"],
                "max_steps": model_args["max_steps"],
                "hist_exog_list": model_args["exog_vars"]
                + model_args["ds_props"],
                "n_series": model_args["n_series"],
                "val_check_steps": model_args["val_check_steps"],
            }
            
            if conf["val_check_steps"] == None: del conf["val_check_steps"]
            #if conf["n_auto_trials"] == None: del conf["n_auto_trials"]
            if conf["encoder_hidden_size"] == None: del conf["encoder_hidden_size"]
            if conf["decoder_hidden_size"] == None: del conf["decoder_hidden_size"]
            if conf["encoder_n_layers"] == None: del conf["encoder_n_layers"]
            if conf["decoder_layers"] == None: del conf["decoder_layers"]
            if conf["context_size"] == None: del conf["context_size"]


            if model_args["model_type"].lower() == "lstm":
                if "n_series" in conf: del conf["n_series"]
                model = LSTM(**conf)
            elif model_args["model_type"].lower() == "gru":
                if "n_series" in conf: del conf["n_series"]
                model = GRU(**conf)
            elif model_args["model_type"].lower() == "rnn":
                if "n_series" in conf: del conf["n_series"]
                model = RNN(**conf)
            elif model_args["model_type"].lower() == "dilatedrnn":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                model = DilatedRNN(**conf)

            elif model_args["model_type"].lower() == "deepar":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                model = DeepAR(**conf)
            elif model_args["model_type"].lower() == "tcn":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                model = TCN(**conf)
            elif model_args["model_type"].lower() == "timesnet":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = TimesNet(**conf)

            elif model_args["model_type"].lower() == "mlp":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = MLP(**conf)
            elif model_args["model_type"].lower() == "nbeats":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = NBEATS(**conf)
            elif model_args["model_type"].lower() == "nbeatsx":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = NBEATSx(**conf)
            elif model_args["model_type"].lower() == "nhits":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = NHITS(**conf)

            elif model_args["model_type"].lower() == "tft":
                if "n_series" in conf: del conf["n_series"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = TFT(**conf)
            elif model_args["model_type"].lower() == "vanillatransformer":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = VanillaTransformer(**conf)
            elif model_args["model_type"].lower() == "informer":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = Informer(**conf)
            elif model_args["model_type"].lower() == "autoformer":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = Autoformer(**conf)
            elif model_args["model_type"].lower() == "fedformer":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = FEDformer(**conf)
            elif model_args["model_type"].lower() == "patchtst":
                if "n_series" in conf: del conf["n_series"]
                if "hist_exog_list" in conf: del conf["hist_exog_list"]
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = PatchTST(**conf)

            elif model_args["model_type"].lower() == "stemgnn":
                if "encoder_hidden_size" in conf: del conf["encoder_hidden_size"]
                if "encoder_n_layers" in conf: del conf["encoder_n_layers"]
                if "decoder_layers" in conf: del conf["decoder_layers"]
                if "context_size" in conf: del conf["context_size"]
                if "decoder_hidden_size" in conf: del conf["decoder_hidden_size"]
                model = StemGNN(**conf)

        if model is not None:
            neural = NeuralForecast(
                models=[model],
                freq=model_args["frequency"],

                local_scaler_type=model_args["local_scaler_type"],
            )

            if model_args.get("crossval", False):
                results_df = neural.cross_validation(training_df)
                model_args["accuracies"] = get_model_accuracy_dict(results_df, r2_score)
            else:
                neural.fit(training_df)

            # persist changes to handler folder
            neural.save(model_args["model_folder"], overwrite=True)
            self.model_storage.json_set("model_args", model_args)

        else:
            logger.error("Model not compiled")

    def predict(self, df, args={}):
        """Makes forecasts with the NeuralForecast Handler.

        NeuralForecast is setup to predict for all groups, so it won't handle
        a dataframe that's been filtered to one group very well. Instead, we make
        the prediction for all groups then take care of the filtering after the
        forecasting. Prediction is nearly instant.
        """
        # Load model arguments
        model_args = self.model_storage.json_get("model_args")

        prediction_df = transform_to_nixtla_df(df, model_args)
        # prediction_df.dtypes = {'y': np.float16}

        groups_to_keep = prediction_df["unique_id"].unique()

        neural = NeuralForecast.load(model_args["model_folder"])
        forecast_df = neural.predict()

        results_df = forecast_df[forecast_df.index.isin(groups_to_keep)].rename(
            {
                "y": model_args["target"],  # auto mode
                "LSTM": model_args["target"],  # non-auto mode
                "GRU": model_args["target"],
                "RNN": model_args["target"],
                "DilatedRNN": model_args["target"],
                "DeepAR": model_args["target"],
                "TCN": model_args["target"],
                "TimesNet": model_args["target"],
                "MLP": model_args["target"],
                "NBEATS": model_args["target"],
                "NBEATSx": model_args["target"],
                "NHITS": model_args["target"],
                "TFT": model_args["target"],
                "VanillaTransformer": model_args["target"],
                "Informer": model_args["target"],
                "Autoformer": model_args["target"],
                "FEDFormer": model_args["target"],
                "PatchTST": model_args["target"],
                "AutoLSTM": model_args["target"],  # non-auto mode
            },
            axis=1,
        )

        return get_results_from_nixtla_df(results_df, model_args)

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
