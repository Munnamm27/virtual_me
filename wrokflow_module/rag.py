from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever



load_dotenv()

def create_vectorstore():
    doccument = PyPDFLoader('Munna Biography.pdf').load()
    embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=500)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, separators=["Chapter", "\n\n", "\n", " ", ""])
    splitted_doccuments = splitter.split_documents(doccument)
    vectorstore = FAISS.from_documents(splitted_doccuments, embedding)


    retriever = vectorstore.as_retriever(search_kwargs={"k":8,})
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    )
    return multi_query_retriever