from flask import Flask, request
app = Flask(__name__)
import sys

@app.route("/", methods=['GET'])
def home():
    return "Home Page: Forecasting API read the documentation to know how to use."

@app.errorhandler(404)
def page_not_found(e):
    return "Forecasting API: The page you're searching for is not found. Read the documentation to know how to use the API"