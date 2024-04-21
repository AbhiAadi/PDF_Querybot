import os
from flask import Flask, render_template, request, jsonify
from query_engine_module import QueryEngine

import json
from llama_index.core import ServiceContext
from llama_index.core import set_global_service_context
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.embeddings.gradient import GradientEmbedding
from llama_index.llms.gradient import GradientBaseModelLLM
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
gradient_access_token = os.getenv('GRADIENT_ACCESS_TOKEN')
gradient_workspace_id = os.getenv('GRADIENT_WORKSPACE_ID')
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('homepage1.html')

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.form['query']
    file = request.files.get('file')

    # Save the uploaded file to a temporary directory
    if file:
        temp_directory = '/tmp/uploads'
        os.makedirs(temp_directory, exist_ok=True)
        file_path = os.path.join(temp_directory, file.filename)
        file.save(file_path)
        print(temp_directory)
        print(file_path)
        query_engine = QueryEngine(temp_directory)

        # response = query_engine.query(user_query)
        # print(response)
        # response = "You asked: {}".format(user_query)
        # print(response)
        response = "{}".format(query_engine.query(user_query))
        # print(response)
    else:
        response = "No file uploaded."


    return jsonify({'response': response})



if __name__ == '__main__':
    app.run(debug=True)
