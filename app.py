# Setup
from ask_about_context import AskAboutContext  
from flask             import Flask, request, jsonify, make_response
from typing            import List
from utils             import delete_files_and_subdirectories
from env import (
    PATH_TO_SAVE_UPLOAD,
    PATH_TO_SAVE_VECTOR_DATABASE,
)

#------------------------------------ Main--------------------------------#
app = Flask(__name__)

# Endpoint - Uploads
@app.route('/uploads', methods=['POST'])
def upload_file(openai_api_key:str) -> str:

    # Check if there is a file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file sent'}), 400

    file = request.files['file']

    # Clean
    delete_files_and_subdirectories(PATH_TO_SAVE_UPLOAD)
    delete_files_and_subdirectories(PATH_TO_SAVE_VECTOR_DATABASE)

    # Save file (context)
    file.save(PATH_TO_SAVE_UPLOAD + "/" + file.filename)

    # Save vector database
    model = AskAboutContext(
        key               = openai_api_key, 
        file_path_context = PATH_TO_SAVE_UPLOAD + "/" + file.filename,
        file_path_db      = PATH_TO_SAVE_VECTOR_DATABASE)
    model.fit() 

    return jsonify({'message': f"Contexto inserido com sucesso {file.filename}"}), 200

# Endpoint - Query
@app.route('/query', methods=['GET'])
def query(openai_api_key:str, query:List[str]) -> List[str]:

    # Load model
    model = AskAboutContext(
        key                      = openai_api_key, 
        file_path_db             = PATH_TO_SAVE_VECTOR_DATABASE,
        there_is_vector_database = True)
    model.fit() 

    # Queries
    answers = [q + ": " + model.query(q) for q in query]

    return make_response(
            jsonify(
                message = "Success!",
                data    = answers,
                status  = 200
            )
        )


app.run(port=5000)
#------------------------------------ Main--------------------------------#