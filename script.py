import os
from flask import Flask, render_template, request
from config import OPENAI_API_KEY

from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

app = Flask(__name__)
app.debug = True
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)

        # Get the user's question
        query = request.form['question']

        # Load the text file
        loader = TextLoader(file_path)

        # Create the index
        index = VectorstoreIndexCreator().from_loaders([loader])

        # Perform the query
        result = index.query(query)

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
