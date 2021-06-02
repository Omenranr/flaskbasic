import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import pmdarima as pmd
import os

def prepare_data(product, data_path):
    '''
    Description:
        this function return a by country splitted dict containing sales,segment,market data.
    parameters:
        product: [String] the product we want to get data for.
        data_path: [String] path to the data.
    outputs:
        prepared_data: [Dict] dictionnary containing data by country for the specific product.
    '''
    try:
        print("READING DATA")
        data = pd.read_csv(os.path.join(data_path, "{0}.csv".format(product)),
                           sep=";",
                           header=0,
                           names=["date", "country", "sales", "market", "segment"])
        print("READ DATA")
        data = data.set_index(pd.to_datetime(data["date"], format="%Y-%m-%d"))
        countries = list(data["country"].unique())
        prepared_data = dict()
        for country in countries:
            prepared_data[country] = data[data["country"] == country]
        return prepared_data
    except FileNotFoundError:
        print("ERROR WHILE READING {0}.csv DATA FILE: FileNotFoundError".format(product.upper()))
    except KeyError:
        print("ERROR WHILE READING {0}.csv DATA FILE: KeyError".format(product.upper()))

def save_models(trained_models, product):
    '''
    Description:
        This is function is responsible of saving the trained model's parameters.
    parameters:
        trained_models: [Dict] dictionnary containing the paramters by country.
        product: [String] product name.
    outputs:
        savec csv paramters files.
    '''
    rows = []
    for country, model in trained_models.items():
        print(model)
        p,q,d = model["order"]
        P,Q,D,S = model["seasonal_order"]
        row = [product, country] + [p,q,d] + [P,Q,D,S]
        rows.append(row)
    model_params = pd.DataFrame(rows, columns=["product", "country", "p", "d", "q", "P", "D", "Q", "S"])
    model_params.to_csv("{0}_model.csv".format(product), sep=";", index=False)
    return model_params

def train_model(prepared_data, product, end):
    '''
    Description:
        This function trains using auto arima a SARIMAX model on the data by country.
    parameters:
        prepare_data: [Dict] dictionnary containing data by country for the specific product.
        end: [String] end date for training.
    outputs:
        models: [Dict] dictionnary of the trained SARIMAX models by country.
    '''
    try:
        models = dict()
        orders = dict()
        for country, data in prepared_data.items():
            #start = data["date"].iloc[0]
            train_data = data.loc[:end]
            print(country)
            sxmodel = pmd.auto_arima(train_data[["sales"]], exogenous=train_data[['segment','market']],
                                       start_p=0, start_q=0,
                                       test='adf',
                                       max_p=3, max_q=3, m=12,
                                       start_P=0, seasonal=True,
                                       d=1, D=1, trace=True, #d=None, D=None, trace=True
                                       error_action='ignore',  
                                       suppress_warnings=True, 
                                       stepwise=True)
            order, seasonal_order = sxmodel.order, sxmodel.seasonal_order
            model = sm.tsa.statespace.SARIMAX(train_data[["sales"]], 
                                              order=order, 
                                              seasonal_order=seasonal_order,
                                              exog=train_data[['segment','market']])
            model = model.fit()
            models[country] = model
            orders[country] = {"order": order, "seasonal_order": seasonal_order}
        _ = save_models(orders, product)
        return models
    except KeyError:
        print("ERROR WHILE TRAINING MODEL: KeyError")
        pass

def load_pretrained(models_path, product, prepared_data, end):
    '''

    '''
    try:
        pretrained_params = pd.read_csv(os.path.join(models_path, "{0}_model.csv".format(product)), sep=";")
        exogenous_params = pd.read_csv(os.path.join(models_path, "{0}_exog.csv".format(product)), sep=";")
        print("Exogenous parameters: ", exogenous_params)
        print("Models parameters: ", pretrained_params)
        #Create statsmodel models
        pretrained_models = dict()
        exogenous_dict = dict()
        grouped_params = pretrained_params.groupby("country")
        grouped_exog = exogenous_params.groupby("country")
        countries = list(pretrained_params["country"].unique())
        for country, data in prepared_data.items():
            #start = data["date"].iloc[0]
            train_data = data.loc[:end]
            params_df = grouped_params.get_group(country)
            exog_params = grouped_exog.get_group(country)
            exog_cols = exog_params.columns[2:]
            exogenous_vars = [e for i,e in zip(exog_params[exog_cols].iloc[0], exog_cols) if i == 1]
            order = list(params_df[["p","d","q"]].iloc[0])
            seasonal_order = list(params_df[["P", "D", "Q", "S"]].iloc[0])
            print(order, seasonal_order, exogenous_vars)
            exogenous_dict[country] = exogenous_vars
            if len(exogenous_vars) == 0:
                model = sm.tsa.statespace.SARIMAX(train_data[["sales"]], 
                                                order=order, 
                                                seasonal_order=seasonal_order)
            else:
                model = sm.tsa.statespace.SARIMAX(train_data[["sales"]], 
                                                order=order, 
                                                seasonal_order=seasonal_order,
                                                exog=train_data[exogenous_vars])
            model = model.fit()
            pretrained_models[country] = model
        return pretrained_models, exogenous_dict, exog_cols
    except FileNotFoundError:
        print("ERROR WHILE LOADING PRETRAINED MODELS: FileNotFoundError {0}".format(models_path))
    except KeyError:
        print("ERROR WHILE LOADING PRETRAINED MODELS: KeyError {0".format(models_path))

def get_predictions(trained_models, prepared_data, end, exog, alpha):
    '''
    Description: 
        this function gives predictions for a specific product.
    parameters:
        product: [String] the name of the product.
        start: [String] start date for training.
        end: [String] end date for training.
        exog: [List] list of exogenous variables to use for forecasting.
        alpha: [Float] confidence risk percentage.
    outputs:
        predictions: [Pandas DataFrame] the predictions by country with mean, min, max.
        coefs: [Pandas DataFrame] the linear regression coefficients.
    '''
    try:
        predictions = dict()
        coefs = dict()
        for country, model in trained_models.items():
            data = prepared_data[country]
            #data.index = pd.to_datetime(data.index)
            forecast_data = data.loc[end:].iloc[1:]
            if len(exog[country]) == 0:
                prediction = model.get_forecast(steps= len(forecast_data)).summary_frame(alpha=alpha)
            else:
                exog_data = forecast_data[exog[country]]
                prediction = model.get_forecast(steps=len(forecast_data), 
                                                          exog=exog_data
                                               ).summary_frame(alpha=alpha)
            prediction.set_index(forecast_data["date"], inplace=True)
            prediction["mean"] = prediction["mean"].clip(lower=0)
            prediction["mean_ci_lower"] = prediction["mean_ci_lower"].clip(lower=0)
            print(country, prediction)
            predictions[country] = prediction
            coefs[country] = model.params
            if "market" in coefs[country]:
                coefs[country]["market"] *= 10_000
        return predictions, coefs
    except KeyError:
        print("ERROR WHILE GENERATING PREDICTIONS: KeyError")
        pass
    except AttributeError:
        print("ERROR WHILE GENERATING PREDICTIONS: NoneType")
        pass

def rectify_conf(temp_df):
    '''
    Description:
        - This function rectifies the confidence interval by applying the rule.
            conf_max = mean + conf[0].
            conf_min = mean - conf[0].
    inputs:
        - temp_df: [DataFrame] dataframe containing the mean, upper_ci, lower_ci.
    outputs:
        - temp_df: [DataFrame] transformed temp_df dataframe.
    '''
    ecart = list((temp_df["mean_ci_upper"] - temp_df["mean_ci_lower"])/2)
    first_ecart = ecart[0]
    temp_df["mean_ci_upper"].iloc[1:] = temp_df["mean"].iloc[1:] + first_ecart
    temp_df["mean_ci_lower"].iloc[1:] = temp_df["mean"].iloc[1:] - first_ecart
    return temp_df

def save_results(output_path, predictions, coefs, product, exog_vars, exog_cols):
    '''
    Description:
        This function takes the predictions and coefs dictionnaries and saves them in the correct format.
    parameters:
        output_path: [String] path where to save results
        predictions: [Dict] dictionnary of the different model's predictions by country.
        coefs: [Dict] dictionnary of the different model's coefs by country.
        exog_vars: [List] list of chosen exogenous variables.
        exog_cols: [List] list of available exogenous variables.
    outputs:
        No outputs.
    '''
    import calendar
    import time
    vol_columns = ["YEAR", "COUNTRY", "CAR_FAMILY", "VOLUME"]
    coef_columns = ["COUNTRY", "CAR_FAMILY", "MARKET", "SEGMENT", "NBCON"]
    prediction_columns = ["mean", "mean_se", "mean_ci_lower", "mean_ci_upper", "country"]
    volumes_df = pd.DataFrame(columns=prediction_columns)
    for country, prediction in predictions.items():
        #VOLUMES PREPARATION
        temp_df = pd.DataFrame(columns=prediction_columns)
        temp_df = pd.concat([temp_df, prediction])
        temp_df.index.rename("year", inplace=True)
        temp_df.index = pd.to_datetime(temp_df.index)
        temp_df = rectify_conf(temp_df)
        temp_df = temp_df.resample('Y').sum()
        temp_df["country"] = [country] * len(temp_df)
        temp_df["date"] = temp_df.index
        temp_df["carfamily"] = [product] * len(temp_df)
        volumes_df = pd.concat([temp_df, volumes_df])
        
    coefs_df = pd.DataFrame(columns=coef_columns)    
    for country, coef in coefs.items():
        #COEFFICIENTS PREPARATING
        exog_list = list(exog_cols)
        print(exog_list)
        exog_coefs = [coef[exog] if exog in exog_vars[country] else 0 for exog in exog_list]
        data_row = [country, product] + exog_coefs + [0]
        temp_df = pd.DataFrame([data_row], columns = coef_columns)
        coefs_df = coefs_df.append(temp_df)
    #FILLING VOLUMES DATAFRAMES
    volumes_df.date = volumes_df.index.year
    IA_TO_PAL_VOLUME = volumes_df[["date", "country", "carfamily", "mean"]]
    IA_TO_PAL_VOLUME_MAX = volumes_df[["date", "country", "carfamily", "mean_ci_upper"]]
    IA_TO_PAL_VOLUME_MIN = volumes_df[["date", "country", "carfamily", "mean_ci_lower"]]
    
    IA_TO_PAL_VOLUME.columns = vol_columns
    IA_TO_PAL_VOLUME_MAX.columns = vol_columns
    IA_TO_PAL_VOLUME_MIN.columns = vol_columns
    print(IA_TO_PAL_VOLUME)
    #SAVING DATAFRAMES:
    ts = str(calendar.timegm(time.gmtime()))
    IA_TO_PAL_VOLUME.to_csv(os.path.join(output_path, "IA_TO_PAL_VOLUMES_{0}_{1}.csv".format(product, ts)), 
                            sep=";", index=False)
    IA_TO_PAL_VOLUME_MAX.to_csv(os.path.join(output_path, "IA_TO_PAL_VOLUMES_MAX_{0}_{1}.csv".format(product, ts)), 
                                sep=";", index=False)
    IA_TO_PAL_VOLUME_MIN.to_csv(os.path.join(output_path, "IA_TO_PAL_VOLUMES_MIN_{0}_{1}.csv".format(product, ts)), 
                                sep=";", index=False)
    coefs_df.to_csv(os.path.join(output_path, "IA_TO_PAL_VOLUMES_COEFF_{0}_{1}.csv".format(product, ts)),
                   sep=";", index=False)
    IA_TO_PAL_VOLUME.reset_index(inplace=True)
    IA_TO_PAL_VOLUME_MAX.reset_index(inplace=True)
    IA_TO_PAL_VOLUME_MIN.reset_index(inplace=True)
    coefs_df.reset_index(inplace=True)
    return IA_TO_PAL_VOLUME, IA_TO_PAL_VOLUME_MAX, IA_TO_PAL_VOLUME_MIN, coefs_df

def main():
    #env_path = "G:\\pmt00\\base\\MTP_P2\\datasources\\IA"
    #parameters_path = env_path + "\\OUT\\PAL_TO_IA_PARAMETERS.csv"
    #output_path = env_path + "\\IN"
    end_date = "2020-12-01"
    env_path = ""
    parameters_path = "../OUT/parameters.csv"
    output_path = "../IN/"
    retrain = False
    alpha = 0.2
    try:
        print("Reading parameters file...")
        parameters = pd.read_csv(parameters_path, names=["value"], sep=";")
        product = str(parameters.loc["product", "value"])
        print("Preparing data...")
        prepared_data = prepare_data(product, env_path)
        if retrain:
            print("Retraining models...")
            trained_models = train_model(prepared_data, end_date)
        else:
            print("Loading pretrained models...")
            trained_models, exog_vars, exog_cols = load_pretrained(product, prepared_data, end_date)
        print("Generating predictions...")
        predictions, coefs = get_predictions(trained_models, prepared_data, end_date, exog_vars, alpha)
        print("Saving results...")
        save_results(output_path, predictions, coefs, product, exog_vars, exog_cols)
    except FileNotFoundError:
        print("ERROR WHILE READING PARAMETERS FILE: FileNotFoundError")
    except KeyError:
        print("ERROR WHILE READING PARAMETERS FILE: KeyError")
    
# if __name__ == "__main__":
#main()