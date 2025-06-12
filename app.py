from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import PredictPipeline 
from src.exception import CustomException
import pandas as pd
import sys

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.html')

@app.route('/predict_review', methods=['POST'])
def predict_review():
    try:
        review_text = request.form['review_text']

        # Provide default values for missing columns
        data = pd.DataFrame({
            'category': ['unknown'],     # or set a valid default like 'electronics'
            'rating': [3],               # assume neutral rating if unknown
            'text_': [review_text]       # rename to match training feature
        })


        pipeline = PredictPipeline()
        prediction = pipeline.predict(data)

        result = "Real" if prediction[0] == 1 else "Fake"
        return render_template('home.html', results=result)

    except Exception as e:
        raise CustomException(e, sys)
    
if __name__ == "__main__":
    app.run(debug=True)