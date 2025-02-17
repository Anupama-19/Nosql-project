from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for using session variables

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Load the trained models
fetal_model = joblib.load("fetal_health_model.pkl")
maternal_model = joblib.load("maternal_health_model.pkl")

# Function to generate and save graphs
# Function to generate and save graphs
import os
import matplotlib.pyplot as plt

def generate_graph(labels, values, title, filename):
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['green', 'red'])
    plt.title(title)
    plt.xlabel("Health Status")
    plt.ylabel("Likelihood (%)")
    
    # Ensure the 'static' directory exists
    os.makedirs('static', exist_ok=True)
    
    graph_path = os.path.join('static', filename)
    plt.savefig(graph_path)  # Save graph
    plt.close()  # Close plot to free memory
    
    return graph_path  # Return graph path


# ----------------- LOGIN SYSTEM -----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "danger")

    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    flash("Logged out successfully!", "info")
    return redirect(url_for('login'))

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        flash("Please log in first.", "warning")
        return redirect(url_for('login'))
    
    return render_template('dashboard.html')

# ----------------- PREDICTION ROUTES -----------------
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            data = [
                float(request.form.get(key, 0)) for key in [
                    'baseline_value', 'accelerations', 'fetal_movement', 'uterine_contractions',
                    'light_decelerations', 'severe_decelerations', 'prolongued_decelerations',
                    'abnormal_short_term_variability', 'mean_value_of_short_term_variability',
                    'percentage_of_time_with_abnormal_long_term_variability', 'mean_value_of_long_term_variability'
                ]
            ]
            data.extend([0] * (21 - len(data)))  # Ensure consistent input size

            prediction = fetal_model.predict([data])[0]
            result = "Normal" if prediction == 1 else "Suspect/Risk"

            # Generate the graph
            graph_path = generate_graph(
                labels=['Normal', 'Suspect/Risk'],
                values=[70 if prediction == 1 else 30, 30 if prediction == 1 else 70],
                title="Fetal Health Prediction",
                filename="prediction_graph.png"
            )

            # Store in session
            session['result'] = result
            session['graph_url'] = graph_path

            return redirect(url_for('result'))  # Redirect to result page

        except Exception as e:
            return render_template('predict.html', result=f"Error: {e}")

    return render_template('predict.html')

@app.route('/result')
def result():
    return render_template('result.html', 
                           result=session.get('result', 'No result available'), 
                           graph_url=session.get('graph_url', None))

# ----------------- MATERNAL HEALTH PREDICTION -----------------
@app.route('/maternal_predict', methods=['POST', 'GET'])
def maternal_predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            data = [
                float(request.form.get(key, 0)) for key in [
                    'age', 'systolicbp', 'diastolicbp', 'bs', 'bodytemp', 'heartrate'
                ]
            ]

            prediction = maternal_model.predict([data])[0]
            result = "Normal" if prediction == 1 else "High Risk"

            graph_path = generate_graph(
                labels=['Normal', 'High Risk'],
                values=[70 if prediction == 1 else 30, 30 if prediction == 1 else 70],
                title="Maternal Health Prediction",
                filename="maternal_prediction_graph.png"
            )

            session['maternal_result'] = result
            session['maternal_graph_url'] = graph_path
            return redirect(url_for('maternal_result'))

        except Exception as e:
            return render_template('maternal_predict.html', result=f"Error: {e}")

    return render_template('maternal_predict.html')

@app.route('/maternal_result')
def maternal_result():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    return render_template('maternal_result.html', 
                           result=session.get('maternal_result', 'No result available'), 
                           graph_url=session.get('maternal_graph_url', None))

# ----------------- CORRELATION ANALYSIS -----------------
@app.route('/correlation_input', methods=['GET', 'POST'])
def correlation_input():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            user_data = {
                "Age": float(request.form['age']),
                "Systolic BP": float(request.form['systolicbp']),
                "Diastolic BP": float(request.form['diastolicbp']),
                "Blood Sugar": float(request.form['bs']),
                "Body Temperature": float(request.form['bodytemp']),
                "Heart Rate": float(request.form['heartrate']),
                "Baseline Fetal HR": float(request.form['baseline_fhr']),
                "Accelerations": float(request.form['accelerations'])
            }

            df = pd.DataFrame([user_data])
            correlation_matrix = df.corr()

            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title("Correlation Heatmap")

            correlation_graph_path = os.path.join('static', 'correlation_graph.png')
            plt.savefig(correlation_graph_path)
            plt.close()

            session['correlation_graph'] = correlation_graph_path
            return redirect(url_for('correlation_result'))

        except Exception as e:
            return render_template('correlation_input.html', error=f"Error: {e}")

    return render_template('correlation_input.html')

@app.route('/correlation_result')
def correlation_result():
    if not session.get('logged_in'):
        return redirect(url_for('login'))

    return render_template('correlation_result.html', 
                           correlation_graph=session.get('correlation_graph', None))

if __name__ == '__main__':
    app.run(debug=True)
