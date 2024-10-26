import os
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI rendering
from flask import Flask, render_template, request, flash
import yfinance as yf
import pandas as pd
from helpers import split_data
from predictors import *
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

timestamp = datetime.now().timestamp()


app = Flask(__name__)
app.secret_key = 'your_secret_key'  


# Directory to save the plots
PLOTS_DIR = "static/plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_prediction", methods=["POST"])
def get_prediction():
    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        period = request.form["period"]

        # Fetch stock data using yfinance
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period)

        if stock_data.empty:
            return """
                <div class="alert alert-danger alert-dismissible fade show" role="alert">
                    Invalid ticker or no data available for the selected period.
                    <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                </div>
            """

        stock_data = stock_data.reset_index()
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data['Date'] = stock_data['Date'].dt.tz_localize(None)
        stock_data = stock_data[["Date", "Close"]]

        # Split the data into training and test sets
        train_data, test_data = split_data(stock_data, period)

        # Determine the number of forecast days
        forecast_days = determine_forecast_days(period)

        # Run Holt-Winters forecast
        hw_test_predictions, hw_future_predictions, hw_model = holtwinters_forecast(
            train_data, test_data, forecast_days, period
        )

        # Run Prophet forecast
        prophet_test_predictions, prophet_future_predictions, prophet_model = prophet_forecast(
            train_data, test_data, forecast_days
        )

        # Generate future dates for the predictions
        last_date = stock_data['Date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        # Prepare the results for display
        results_hw = pd.DataFrame({
            "Date": future_dates,
            "HoltWinters_Predicted_Close": hw_future_predictions
        })

        results_prophet = pd.DataFrame({
            "Date": future_dates,
            "Prophet_Predicted_Close": prophet_future_predictions
        })

        # Merge results to display them side by side
        merged_results = pd.merge(results_hw, results_prophet, on="Date")

        # Evaluate both models on the test data
        hw_evaluation = evaluate_model(test_data, hw_test_predictions)
        prophet_evaluation = evaluate_model(test_data, prophet_test_predictions)

        # Plot the results for both methods
        plot_file_hw = os.path.join(PLOTS_DIR, f"{ticker}_holtwinters.png")
        plot_file_prophet = os.path.join(PLOTS_DIR, f"{ticker}_prophet.png")
        plot_predictions(stock_data, train_data, test_data, hw_test_predictions, results_hw, "Holt-Winters Prediction vs Actual", plot_file_hw)
        plot_predictions(stock_data, train_data, test_data, prophet_test_predictions, results_prophet, "Prophet Prediction vs Actual", plot_file_prophet)

        # Render the partial result HTML for HTMX
        return render_template(
            "result.html",
            ticker=ticker,
            hw_evaluation=hw_evaluation,
            prophet_evaluation=prophet_evaluation,
            plot_hw=plot_file_hw,
            plot_prophet=plot_file_prophet,
            timestamp=timestamp
        )



def plot_predictions(stock_data, train_data, test_data, test_predictions, prediction_data, title, save_path):
    """Plot actual stock data with train, test, and future predictions."""
    plt.figure(figsize=(14, 8))  # Increase the canvas size

    # Plot the training data
    plt.plot(train_data['Date'], train_data['Close'], label="Train Data", color="blue", linewidth=2)

    # Plot the actual test data
    plt.plot(test_data['Date'], test_data['Close'], label="Actual Test Data", color="purple", linewidth=2)

    # Plot the test predictions with markers
    plt.plot(test_data['Date'], test_predictions, label="Test Predictions", linestyle="--", color="orange", marker='o', linewidth=2)

    # Plot the future predictions with a dotted line
    plt.plot(prediction_data['Date'], prediction_data.iloc[:, 1], label="Future Predictions", linestyle=":", color="green", linewidth=2)

    # Adding plot title and labels
    plt.title(title, fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Stock Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid()

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    app.run(debug=True)
