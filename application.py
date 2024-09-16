from flask import Flask, request, render_template
import pickle as pkl
import numpy as np

application = Flask(__name__)
app = application

# Load model and scaler pickle files
model = pkl.load(open('./model/Model.pkl', 'rb'))
scaler = pkl.load(open('./model/Scaler.pkl', 'rb'))

# Define min and max values for normalization (update these with your data's range)
MIN_WIN_POINTS = 0
MAX_WIN_POINTS = 10000000000  # Example maximum value, update with your actual max

@app.route("/", methods=['GET', 'POST'])
def predict_data():
    if request.method == 'POST':
        try:
            boosts = float(request.form.get('boosts'))
            players_knocked = float(request.form.get('players_knocked'))
            kill_rank = float(request.form.get('kill_rank'))
            revives = float(request.form.get('revives'))
            rankPoints = float(request.form.get('rankPoints'))
            matchDuration = float(request.form.get('matchDuration'))
            longestKill = float(request.form.get('longestKill'))
            kills = float(request.form.get('kills'))
            winPoints = float(request.form.get('winPoints'))
            walkDistance = float(request.form.get('walkDistance'))

            # Transform and predict
            new_scaled_data = scaler.transform([[boosts, players_knocked, kill_rank, revives, rankPoints, matchDuration, longestKill, kills, winPoints, walkDistance]])
            result = model.predict(new_scaled_data)[0]

            # Normalize the result to percentage
            result_percentage = max(0, min(100, (result - MIN_WIN_POINTS) / (MAX_WIN_POINTS - MIN_WIN_POINTS) * 100))

            return render_template('home.html', results=f"{result_percentage:.2f}%")

        except Exception as e:
            print(f"Error: {e}")
            return render_template('home.html', results='Error occurred during prediction.')

    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
