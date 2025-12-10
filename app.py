from flask import Flask, render_template, jsonify, request
# from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# from src.prompt import *
import os 




app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')

if __name__ == "__main__":
    app.run(debug=True)   # <-- REQUIRED
