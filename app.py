from flask import Flask, request, jsonify, send_from_directory
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_name = "finetunedmodel2"  # Path to your fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

@app.route('/')
def index():
    return send_from_directory('.', 'index2.html')

@app.route('/generate', methods=['POST'])
def generate():
    user_input = request.json.get('prompt')
    if not user_input:
        return jsonify({"error": "No prompt provided"}), 400

    inputs = tokenizer.encode(user_input, return_tensors='pt')
    

         # Generate the attention mask
    attention_mask = (inputs != tokenizer.pad_token_id).long()
    max_length=1000
    # Generate the text
    output = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
