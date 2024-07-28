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


llm = Ollama(model=model_name, base_url='https://68e3-103-241-244-6.ngrok-free.app')



def get_ai_response(prompt):
    response = llm.invoke(prompt)

    print(response)

    return response