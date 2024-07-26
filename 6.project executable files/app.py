import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# Load the pickled model
model = pickle.load(open('airnbnb model.pkl','rb'))

# Create the Flask application
app = Flask(__name__)

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/portfolio.html')
def portfolio():
    return render_template('portfolio.html')

@app.route('/pricing.html')
def pricing():
    return render_template('pricing.html')

@app.route('/predict.html', methods=['POST', 'GET'])
def predict():
    return render_template('predict.html')
    if request.method == 'POST':
        Hotelid = int(request.form['Hotelid'])
        Property_type = int(request.form['property_type'])
        Room_type = int(request.form['room_type'])
        Accommodates = int(request.form['accommodates'])
        Cancellation_policy = int(request.form['cancellation_policy'])
        Cleaning_fee = int(request.form['cleaning_fee'])  # Corrected variable name
        City = int(request.form['city'])
        Host_identity_verified = int(request.form['host_identity_verified'])
        Instant_bookable = int(request.form['instant_bookable'])
        Latitude = float(request.form['latitude'])
        Longitude = float(request.form['longitude'])
        Number_of_reviews = int(request.form['number_of_reviews'])
        Bedrooms = int(request.form['bedrooms'])

        # Make prediction
        prediction = model.predict(pd.DataFrame([[Hotelid, Property_type, Room_type, Accommodates, Cancellation_policy,
                                                  Cleaning_fee, City, Host_identity_verified, Instant_bookable,
                                                  Latitude, Longitude, Number_of_reviews, Bedrooms]]))
        prediction = np.round(prediction, 4)

        return render_template('predict.html', prediction_text="is {}".format(prediction))
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
