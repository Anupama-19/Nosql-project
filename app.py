from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using session variables

# Load the trained model
model = joblib.load("fetal_health_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Collect input data from the form
            data = [
                float(request.form.get('baseline_value', 0)),
                float(request.form.get('accelerations', 0)),
                float(request.form.get('fetal_movement', 0)),
                float(request.form.get('uterine_contractions', 0)),
                float(request.form.get('light_decelerations', 0)),
                float(request.form.get('severe_decelerations', 0)),
                float(request.form.get('prolongued_decelerations', 0)),
                float(request.form.get('abnormal_short_term_variability', 0)),
                float(request.form.get('mean_value_of_short_term_variability', 0)),
                float(request.form.get('percentage_of_time_with_abnormal_long_term_variability', 0)),
                float(request.form.get('mean_value_of_long_term_variability', 0)),
                *([0] * (21 - 11))  # Fill missing values with 0 if fewer inputs are provided
            ]

            # Make prediction
            prediction = model.predict([data])[0]
            result = "Normal" if prediction == 1 else "Suspect/Risk"

            # Generate a simple bar graph
            labels = ['Normal', 'Suspect/Risk']
            values = [70 if prediction == 1 else 30, 30 if prediction == 1 else 70]

            fig, ax = plt.subplots()
            ax.bar(labels, values, color=['green', 'red'])
            plt.title("Prediction Result")
            plt.xlabel("Fetal Health Status")
            plt.ylabel("Likelihood (%)")
            graph_path = os.path.join('static', 'prediction_graph.png')
            plt.savefig(graph_path)
            plt.close()

            # Store result and graph path in session
            session['result'] = result
            session['graph_url'] = graph_path

            # Redirect to result page
            return redirect(url_for('result'))

        except Exception as e:
            return render_template('predict.html', result=f"Error: {e}")

    return render_template('predict.html')

@app.route('/result')
def result():
    # Fetch result and graph URL from session
    result = session.get('result', 'No result available')
    graph_url = session.get('graph_url', None)
    return render_template('result.html', result=result, graph_url=graph_url)

if __name__ == '__main__':
    app.run(debug=True)
