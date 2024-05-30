from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model from the file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template("forest.html", pred=None, bhai=None)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final = [np.array(features)]
    prediction = model.predict(final)
    output = prediction[0]
    
    # Calculate accuracy (this is a placeholder; replace with your actual accuracy calculation if needed)
    accuracy = 0.85  # Assume a fixed accuracy for demonstration purposes

    return render_template('forest.html', pred=f'The temperature should be {output:.2f}°C or higher.', bhai=f'Accuracy: {accuracy * 100:.2f}%')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
