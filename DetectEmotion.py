from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load the model and vectorizer
with open('emotion_detection_model.pkl', 'rb') as f:
  model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
  vectorizer = pickle.load(f)

@app.route('/')
def index():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
  # Get the text from the request
  text = request.form.get('text')

  # Vectorize the text
  text_vec = vectorizer.transform([text])

  # Make a prediction
  prediction = model.predict(text_vec)[0]

  # Return the prediction as a JSON response
  return jsonify({'prediction': prediction})

if __name__ == '__main__':
  app.run(debug=True)
