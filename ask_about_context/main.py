
# Setup
from langchain.text_splitter             import RecursiveCharacterTextSplitter # Split - Tokenization
from langchain.embeddings                import OpenAIEmbeddings               # Embeddings
from langchain.vectorstores              import Chroma                         # Vector Store
from langchain.llms                      import OpenAI                         # Models
from langchain.chat_models               import ChatOpenAI                     # Chat
from langchain.chains.question_answering import load_qa_chain                  # QA
from langchain.callbacks                 import get_openai_callback            # Callback
import os
import textract
import warnings
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None) 

class AskAboutContext:

    def __init__(self, 
                 key: str, 
                 file_path_context: str, 
                 embedding_model_name:str = "text-embedding-ada-002", 
                 model_name:str           = "gpt-3.5-turbo") -> None:
        """_summary_

        :param key: Chave de acesso a openAI
        :type key: str

        :param file_path_context: _description_
        :type file_path_context: Path de onde está o arquivo de contexto

        :param embedding_model_name: Modelo que realizará o embeeding, defaults to "text-embedding-ada-002"
        :type embedding_model_name: str, optional

        :param model_name: Modelo de LLM que responderá as questões, defaults to "gpt-3.5-turbo"
        :type model_name: str, optional

        """
        self.__key                  = key
        self.__file_path_context    = file_path_context  
        self.__embedding_model_name = embedding_model_name 
        self.__model_name           = model_name 
        
        
    def __document_loading(self) -> str:
        """_summary_

        :return: Arquivo de texto no formato de string
        :rtype: str
        """

        file_path = self.__file_path_context
        doc       = textract.process(file_path)
        text      = doc.decode('utf-8')

        return text
    

    def __tokenization(self) -> list:
        """_summary_
        Realiza a tokenização

        :return: Uma lista de documentos, onde cada documento é um segmento de texto que foi tokenizado
        :rtype: list
        """

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size      = 512, # Quantidade máxima de caracteres por split
            chunk_overlap   = 24,  # Quantidade de máxima de caracteres sobrepostos por split
        )
        chunks = text_splitter.create_documents([self.__document_loading()])

        return chunks
        
        
    def __embeddings(self) -> Chroma:
        """_summary_

        :return: Espaço de vetores
        :rtype: Chroma
        """

        embeddings = OpenAIEmbeddings(
            openai_api_key = self.__key , 
            model          = self.__embedding_model_name
        )
        vectordb = Chroma.from_documents(
            documents         = self.__tokenization(), 
            embedding         = embeddings,
            )

        return vectordb 


    def fit(self) -> None:
        """_summary_
        """
        self.__vectordb  = self.__embeddings()

    def query(self, query:str) -> str:
        """_summary_

        :param query: Perguntas sobre o contexto
        :type query: str
        :return: Resposta sobre as perguntas
        :rtype: str
        """
        # Retrieval
        # Maximum Marginal Relevance(MMR)
        docs  = self.__vectordb.max_marginal_relevance_search(
            query, 
            k      = 2 , 
            fetch_k= 5)

        # Question Answering
        llm   = ChatOpenAI(openai_api_key = self.__key, model_name = self.__model_name, temperature = 0)
        chain = load_qa_chain(llm, chain_type="stuff")

        with get_openai_callback() as cb:
            return chain.run(input_documents = docs, question = query)
    
