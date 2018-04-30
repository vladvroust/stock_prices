import csv
import numpy as np
from sklearn.svm import SVR as svr
import matplotlib.pyplot as plt
#import pandas as pd

#initialise the data
dates = []
prices = []

#plt.switch_backend('newbackend')

#get stock price
#def get_historical(quote):
#    # Download our file from google finance
#    url = 'http://www.google.com/finance/historical?q='+quote+'&output=csv'
#    r = requests.get(url, stream=True)
#
#    if r.status_code != 400:
#        with open(FILE_NAME, 'wb') as f:
#            for chunk in r:
#                f.write(chunk)
#
#    return True

#get prices for each days

def get_data(filename):
	with open(filename, "r") as csv_file:
		cvsFileReader = csv.reader(csv_file, delimiter = "\t")
		next(cvsFileReader) #skip first row
		for row in cvsFileReader:
			dates.append(int(row[0].split('/')[0])) #date of the month
			prices.append(float(row[4]))
	return

def predict_prices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	svr_lin = svr(kernel = "linear", C = 1e3)
	svr_poly = svr(kernel = "poly", C = 1e3, degree = 2)
	svr_rbf = svr(kernel = "rbf", C = 1e3, gamma = 0.1)

	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)
	svr_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color="black", label="Data")
	plt.plot(dates, svr_lin.predict(dates), color = "red", label = "Linear Model")
	plt.plot(dates, svr_poly.predict(dates), color = "green", label = "Polynomial Model")
	plt.plot(dates, svr_rbf.predict(dates), color = "blue", label = "RBF Model")
	plt.xlabel("Date")
	plt.ylabel("Price")
	plt.title("Support Vector Regression")
	plt.legend()
	plt.show()

	return svr_lin.predict(x)[0], svr_poly.predict(x)[0], svr_rbf.predict(x)[0]

#get dataset on boursorama with two values: dates and prices
#https://www.boursorama.com/cours/1rPMC/ 1M
get_data('LVMHMOETVUITTON_2018-04-30.txt')

#predict price on a set day x
predicted_price = predict_prices(dates, prices, 29)
print(predicted_price)





