# vegetation_forecasts
A repository for testing ecological forecasting methods of satellite derived vegetation indices

## Process

1. Select test location
2. Calculate NDVI anomalies with smoothed timeseries.
3. Check for autoregression potential in vegetation anomalies timeseries
   * Time series plot of several pixels
   * Lagplot
     * This plots the observation at previous time step with the observation at the next time step as a scatter plot.
   * Check pearson correlation coefficient between lagged time steps using `df.corr()` and `pandas.plotting import autocorrelation_plot`
   * Check the mean over time of the timeseries, it should be relatively stationary.
4. The data must be "stationary". Remove seasonality? 
5. autoregression
   * Train / Test split
   * `from statsmodels.tsa.ar_model import AR`
   * `model = AR(NDVI_anomalies_train)`
   * `model_fitted = model.fit()`

## Notes and resources
* [Forecasting vegetation condition for drought early warning systems in pastoral communities in Kenya](https://www.sciencedirect.com/science/article/pii/S003442572030256X)
* Linear autoregression formula: `y = a + b1*X(t-1) + b2*X(t-2) + b3*X(t-3)`
* [Example forecasting in Python](https://pythondata.com/forecasting-time-series-autoregression/)
* [Autoregression Models for Time Series Forecasting With Python](https://machinelearningmastery.com/autoregression-models-time-series-forecasting-python/)
* [ScitKit-learn-Inspired Time Series models](https://github.com/EthanRosenthal/skits)
  * [Applied example](https://www.ethanrosenthal.com/2018/03/22/time-series-for-scikit-learn-people-part2/)
  
* https://www.machinelearningplus.com/time-series/time-series-analysis-python/

[ml-forecast]
https://nixtla.github.io/mlforecast/forecast.html
