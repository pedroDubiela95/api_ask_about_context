# Setup
from langchain.text_splitter             import RecursiveCharacterTextSplitter # Split - Tokenization
from langchain.embeddings                import OpenAIEmbeddings               # Embeddings
from langchain.vectorstores              import Chroma                         # Vector Store
from langchain.llms                      import OpenAI                         # Models
from langchain.chat_models               import ChatOpenAI                     # Chat
from langchain.chains.question_answering import load_qa_chain                  # QA
from langchain.callbacks                 import get_openai_callback            # Callback
from typing                              import List
from langchain_core.documents.base       import Document

import os
import pandas as pd
import textract
import warnings
warnings.filterwarnings("ignore")


class AskAboutContext:
    """
    A class that executes a complete pipeline to ask questions 
    about a specific context using ChatGPT.
    """

    def __init__(self, 
                 key:str, 
                 file_path_db:str,
                 file_path_context:str    = None,
                 embedding_model_name:str = "text-embedding-ada-002", 
                 model_name:str           = "gpt-3.5-turbo",
                 there_is_vector_database = False) -> None:
        
        """
        Initializes an instance of the AskAboutContext class.

        :param key: OpenAI API key
        :type key: str
        :param file_path_context: Path to the context file
        :type file_path_context: str
        :param file_path_db: Path to save the vector database
        :type file_path_db: str
        :param embedding_model_name: Name of the embedding model to perform embedding, defaults to "text-embedding-ada-002"
        :type embedding_model_name: str, optional
        :param model_name: Name of the LLM model that will respond to questions, defaults to "gpt-3.5-turbo"
        :type model_name: str, optional
        :param there_is_vector_database: If already there is a vector database in disk
        :type there_is_vector_database: boolean, optional
        """

        self.__key                      = key
        self.__file_path_context        = file_path_context
        self.__file_path_db             = file_path_db   
        self.__embedding_model_name     = embedding_model_name 
        self.__model_name               = model_name 
        self.__there_is_vector_database = there_is_vector_database 
        
        
    def __document_loading(self) -> str:
        """
        Loads the context file into memory and converts it to a string.

        :return: The context in string format
        :rtype: str
        """

        file_path = self.__file_path_context
        doc       = textract.process(file_path)
        text      = doc.decode('utf-8')

        return text
    

    def __tokenization(self) -> List[Document]:
        """
        Performs the splitting process of the documents.

        :return: A list with the context divided into several documents
        :rtype: List[Document]
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size      = 512, # Maximum number of characters per split.
            chunk_overlap   = 24,  # Maximum number of overlapping characters per split.
        )
        chunks = text_splitter.create_documents([self.__document_loading()])
        return chunks
        
        
    def __embeddings(self) -> Chroma:
        """
        Performs the embedding process of each chunk resulting from tokenization.
        In other words, it creates a numerical vector representation of each chunk.
        Finally, it stores these embeddings and their respective chunks in a database, 
        using the vectors themselves as indexes, resulting in an embedding space.

        When the vector database already exists on disk, it just loads it

        :return: Vector space
        :rtype: Chroma
        """

        self.__embedding_funtion = OpenAIEmbeddings(
            openai_api_key = self.__key , 
            model          = self.__embedding_model_name
            )
        
        persist_directory = self.__file_path_db 
        
        # Load    
        if self.__there_is_vector_database:
            vectordb = Chroma(
                persist_directory  = persist_directory,
                embedding_function = self.__embedding_funtion,
            )

        # Create
        else:
            vectordb = Chroma.from_documents(
                documents         = self.__tokenization(), 
                embedding         = self.__embedding_funtion,
                persist_directory = persist_directory)

        return vectordb 


    def fit(self) -> None:
        """
        Executes all the necessary pipeline steps to have the vector space ready.
        """
        self.__vectordb = self.__embeddings()


    def query(self, query:str) -> str:
        """
        Executes the following pipeline:
        1. User asks a question (query) about the context.
        2. This question also undergoes an embedding process (using the same model as the context).
        3. The chunks most similar to the query (distance between vectors) are selected from the vector database.
        4. The ChatGPT model is informed of the query and only the most relevant documents to answer it.

        :param query: A question about the context
        :type query: str
        :return: The answer to the question
        :rtype: str
        """

        docs  = self.__vectordb.max_marginal_relevance_search(
            query, 
            k      = 2, 
            fetch_k= 5)

        # Question Answering
        llm   = ChatOpenAI(openai_api_key = self.__key, model_name = self.__model_name, temperature = 0)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            return chain.run(input_documents = docs, question = query)
        
if __name__ == "__main__":

    # Constants
    OPENAI_API_KEY               = os.environ.get("OPENAI_API_KEY") 
    PATH_TO_SAVE_UPLOAD          = os.environ.get("PATH_TO_SAVE_UPLOAD") 
    PATH_TO_SAVE_VECTOR_DATABASE = os.environ.get("PATH_TO_SAVE_VECTOR_DATABASE") 
    PATH_TO_SAVE_MODEL           = os.environ.get("PATH_TO_SAVE_MODEL") 

    # Create vector database
    model = AskAboutContext(
        key               = OPENAI_API_KEY, 
        file_path_context = PATH_TO_SAVE_UPLOAD + "/" + "precos.jpg",
        file_path_db      = PATH_TO_SAVE_VECTOR_DATABASE)
    model.fit() 

    model.query("Qual modelo é mais caro?")

    # Load vector databse
    model2 = AskAboutContext(
        key               = OPENAI_API_KEY, 
        file_path_db      = PATH_TO_SAVE_VECTOR_DATABASE,
        there_is_vector_database=True)
    model2.fit() 

    model2.query("Qual modelo é mais barato?")