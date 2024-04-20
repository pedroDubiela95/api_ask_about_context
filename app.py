# Setup
from flask             import Flask, request, jsonify
from ask_about_context import AskAboutContext  
import os
import pickle

# Constants
OPENAI_API_KEY               = os.environ.get("OPENAI_API_KEY") 
PATH_TO_SAVE_UPLOAD          = os.environ.get("PATH_TO_SAVE_UPLOAD") 
PATH_TO_SAVE_VECTOR_DATABASE = os.environ.get("PATH_TO_SAVE_VECTOR_DATABASE") 
PATH_TO_SAVE_MODEL           = os.environ.get("PATH_TO_SAVE_MODEL") 

app = Flask(__name__)

@app.route('/uploads', methods=['POST'])
def upload_file(openai_api_key:str) -> str:

    # Check if there is a file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent'}), 400

    file = request.files['file']

    # Save file (context)
    file.save(PATH_TO_SAVE_UPLOAD + "/" + file.filename)

    # chatgpt
    model = AskAboutContext(
        key               = OPENAI_API_KEY, 
        file_path_context = PATH_TO_SAVE_UPLOAD + "/" + file.filename,
        file_path_db      = PATH_TO_SAVE_VECTOR_DATABASE)
    model.fit() 

    
    # Salvando a classe como um arquivo pickle
    with open(PATH_TO_SAVE_MODEL + "/model.pkl", "wb") as file_pickle:
        pickle.dump(model, file_pickle)


    # Return
    return jsonify({'message': f"Contexto inserido com sucesso {file.filename}"}), 200

app.run(port=5000)
