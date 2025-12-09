# from langchain.chains.combine_documents import StuffDocumentsChain
# from langchain_community.document_combiners import StuffDocumentsChain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains import create_retrieval_chain
from langchain_classic.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os
import helper


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)


embedding = helper.download_embeddings()

## Load from existing index
index_name = "medical-assistance-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name,
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})



model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

system_prompt = (
    "You are a medical assistant for question-answering with the patient."
    "\n\n"
    "Below is the RAG obtained context that might help:"
    "{context}"
    
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(model, prompt)

rag_chain = create_retrieval_chain(
    retriever, question_answer_chain
)


response = rag_chain.invoke({"input": "what is are the symptoms of diabetes? explain in detail"})
print(response["answer"])
