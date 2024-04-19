# Setup
from flask import Flask, request, jsonify
import os

# Constants
PATH_TO_SAVE_FILE_FROM_UPLOAD = os.environ.get("PATH_TO_SAVE_FILE_FROM_UPLOAD") 
print(PATH_TO_SAVE_FILE_FROM_UPLOAD)


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file(openai_api_key:str) -> str:

    # Check if there is a file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent'}), 400

    file = request.files['file']

    # Save file
    filename = PATH_TO_SAVE_FILE_FROM_UPLOAD + "/" +file.filename
    file.save(filename)

    # chatgpt

    # Remove file
    # Return
    return jsonify({'message': f"Arquivo PDF enviado com sucesso {file.filename}"}), 200

app.run(port=5000)
