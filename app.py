from flask import Flask, request
app = Flask(__name__)
from predict_sales_v1 import *
import sys

@app.route("/", methods=['GET'])
def home():
    return "Home Page: Forecasting API read the documentation to know how to use."

@app.route("/forecast/<product>", methods=['GET'])
def forecast(product):
    '''
    Description: forecasting route taking as argument the product to forecast and do it for countries.
    '''
    end_date = "2020-12-01"
    retrain = bool(request.args.get("retrain", False))
    alpha = float(request.args.get("alpha", 0.2))
    data_path = "./data/"
    models_path = "./models/"
    output_path = "./forecasts/"
    prepared_data = "EMPTY"
    trained_models = "EMPTY"
    wmlActive = bool(request.args.get('wml'), True)
    try:
        print("Preparing Data...")
        prepared_data = prepare_data(product, data_path)
        if retrain:
            print("Retraining Models...")
            trained_models = train_model(prepared_data, end_date)
        else:
            print("Using Pretrained Models...")
            trained_models, exog_vars, exog_cols = load_pretrained(models_path, product, prepared_data, end_date)
        print("Generating predictions...")
        predictions, coefs = get_predictions(trained_models, prepared_data, end_date, exog_vars, alpha)
        print("Saving results...")
        IA_TO_PAL_VOLUME, IA_TO_PAL_VOLUME_MAX, IA_TO_PAL_VOLUME_MIN, coefs_df = save_results(output_path, predictions, coefs, product, exog_vars, exog_cols)
        print("Returning results...")
        if wmlActive:
            return {
                "values": "23618.026074508816,23791.67804542294,23860.984103128325,23991.93496957897,24086.4677116548,24202.515319458584,2837.517784580972,3065.9195997998195,3745.516942754864,4059.8873458016715,4487.627145984945,4801.637850588014,18056.593411238227,29179.458737779405,17782.58605031984,29800.770040526044,16519.90579184422,31202.06241441243,16034.701990517784,31949.16794864016,15290.880129480038,32882.055293829566,14791.478065501758,33613.55257341541", 
                "errors": "errors"
                }
        return {
            "volumes": IA_TO_PAL_VOLUME.to_json(),
            "ci_max": IA_TO_PAL_VOLUME_MAX.to_json(),
            "ci_min": IA_TO_PAL_VOLUME_MIN.to_json(),
            "coefs": coefs_df.to_json()
        }
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise

@app.route("/forecast/wml/<product>", methods=['GET'])
def wml_mimic_forecast(product):
    return

@app.route("/train/<product>", methods=["POST"])
def train_model(product):
    return "training model route"

@app.errorhandler(404)
def page_not_found(e):
    return "Forecasting API: The page you're searching for is not found. Read the documentation to know how to use the API"