# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
os.environ['USER_AGENT'] = 'RagApp/1.0'
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.llms.ollama import Ollama

model_name = 'qwen2:1.5b-instruct'

llm = Ollama(model=model_name, base_url='https://0907-2402-a00-401-7324-b419-dfbf-7153-dbbc.ngrok-free.app')

model_name = "jinaai/jina-embeddings-v2-base-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load, chunk and index the contents of the blog.
# loader = WebBaseLoader(web_paths=('https://www.utu.ac.in/AMTICS/index.html', 'https://www.utu.ac.in/AMTICS/AboutUs.html', 'https://www.utu.ac.in/AMTICS/Laboratory.html', 'https://www.utu.ac.in/AMTICS/staff.html'))
loader = WebBaseLoader(web_paths=('https://www.utu.ac.in/AMTICS/AboutUs.html','https://www.utu.ac.in/AMTICS/staff.html'))

docs = loader.load()
print(docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

rag_prompt_template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""
prompt = PromptTemplate.from_template(rag_prompt_template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


# answer = rag_chain.invoke("Who is the director of the AMTICs?")

# print("================\n")
# print(answer)