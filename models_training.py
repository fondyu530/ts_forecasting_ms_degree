import pandas as pd
import datetime as dt
import plotly.express as px

from tqdm import tqdm
from sklearn.metrics import (mean_squared_error as mse, mean_absolute_error as mae,
                             mean_absolute_percentage_error as mape)


def ts_forecasting_cv(PredictorClass, ts_data, start_date, end_date, predictor_description,
                      step=14, forecasts_num=14, trace=False):
    errors_dict = {"test_num": [], "rmse": [], "mae": [], "mape": [], "bias": [],
                   "rmse_corrected": [], "mae_corrected": [], "mape_corrected": []}

    with tqdm(total=int((start_date - end_date) / dt.timedelta(1) / step)) as pbar:
        i = 1
        while start_date + dt.timedelta(step) <= end_date:
            ts_train = ts_data[ts_data["date"] < start_date].copy()
            ts_test = ts_data[(ts_data["date"] >= start_date) & \
                              (ts_data["date"] < start_date + dt.timedelta(step))].copy().reset_index(drop=True)

            predictor = PredictorClass(predictor_description, ts_train)
            ts_forecast = predictor.predict_ts(forecasts_num).reset_index(drop=True)

            rmse_err = mse(ts_test["value"], ts_forecast["value"], squared=False)
            mae_err = mae(ts_test["value"], ts_forecast["value"])
            mape_err = mape(ts_test["value"], ts_forecast["value"])

            if trace:
                ts_test["type"] = "actual"
                ts_forecast["type"] = "forecast"
                ts_plot = pd.concat([ts_test, ts_forecast])

                fig = px.line(ts_plot, x="date", y="value", color="type")
                fig.show()

            mask = ~ts_test["date"].dt.weekday.isin((5, 6))
            ts_test, ts_forecast = ts_test[mask], ts_forecast[mask]

            rmse_err_corrected = mse(ts_test["value"], ts_forecast["value"], squared=False)
            mae_err_corrected = mae(ts_test["value"], ts_forecast["value"])
            mape_err_corrected = mape(ts_test["value"], ts_forecast["value"])

            errors_dict["rmse"].append(rmse_err)
            errors_dict["mape"].append(mape_err)
            errors_dict["mae"].append(mae_err)
            errors_dict["bias"].append(round(ts_forecast["value"].sum() / ts_test["value"].sum(), 2))
            errors_dict["rmse_corrected"].append(rmse_err_corrected)
            errors_dict["mae_corrected"].append(mae_err_corrected)
            errors_dict["mape_corrected"].append(mape_err_corrected)
            errors_dict["test_num"].append(i)

            start_date += dt.timedelta(step)
            i += 1
            pbar.update(1)

    return pd.DataFrame(errors_dict)