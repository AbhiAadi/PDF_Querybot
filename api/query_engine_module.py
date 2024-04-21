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

# Now you can use these variables in your application

class Backend:
    def __init__(self):
        self.llm = None
        self.embed_model = None
        self.service_context = None

    def init(self):
      # self.setup_environment_variables()
        self.setup_llm()
        self.setup_embed_model()
        self.setup_service_context()

    def setup_llm(self):
        self.llm = GradientBaseModelLLM(
            base_model_slug="llama2-7b-chat",
            max_tokens=400,
        )

    def setup_embed_model(self):
        self.embed_model = GradientEmbedding(
            gradient_access_token = gradient_access_token,
            gradient_workspace_id = gradient_workspace_id,
            gradient_model_slug="bge-large",
        )

    def setup_service_context(self):
        self.service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model,
            chunk_size=256,
        )

        set_global_service_context(self.service_context)


class QueryEngine:
    def __init__(self, pdf_directory):
        backend = Backend()
        backend.init()
        self.load_documents(pdf_directory)
        self.create_index()

    def load_documents(self, pdf_directory):
        self.documents = SimpleDirectoryReader(pdf_directory).load_data()

    def create_index(self):
        self.index = VectorStoreIndex.from_documents(self.documents)

    def query(self, user_query):
        return self.index.as_query_engine().query(user_query)
