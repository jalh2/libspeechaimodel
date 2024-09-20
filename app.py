from flask import Flask, request, jsonify, send_from_directory
import requests
import os

app = Flask(__name__)

# Hugging Face API information
model_endpoint = "https://api-inference.huggingface.co/models/huggingjallah/libspeechmodel"  # Replace with your model
api_token = "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"  # Replace with your Hugging Face API token
headers = {"Authorization": f"Bearer {api_token}"}

@app.route('/')
def index():
    return send_from_directory('.', 'index2.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('prompt')
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    # Payload to send to the Hugging Face model API
    payload = {"inputs": user_input}
    
    # Send POST request to Hugging Face Inference API
    response = requests.post(model_endpoint, headers=headers, json=payload)
    
    if response.status_code != 200:
        return jsonify({"error": "Model request failed"}), response.status_code

    # Parse the response from the API
    generated_text = response.json().get("generated_text", "")
    
    return jsonify({"response": generated_text})

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

