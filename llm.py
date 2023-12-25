import os
import pinecone
import environ
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DataFrameLoader
from dataManip import resultingInpatientStrgDataJSON
import replicate

env = environ.Env()
environ.Env.read_env()

REPLICATE_API_TOKEN = env('REPLICATE_API_TOKEN')
PINECONE_API_KEY = env("PINECONE_API_KEY")

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

# loader = DataFrameLoader(resultingInpatientStrgData, page_content_column="sentence")
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

pinecone.init(      
	api_key=PINECONE_API_KEY,      
	environment='gcp-starter'      
)      
index_name = "stringeddata"
index = pinecone.Index(index_name)
# vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

vectordb = Pinecone.from_existing_index("stringeddata", embeddings)

# model training
training = replicate.trainings.create(
  version="meta/llama-2-7b:73001d654114dad81ec65da3b834e2f691af1e1526453189b7bf36fb3f32d0f9",
  input={
    "train_data": "https://gist.github.com/kevinlu625/e36d6efef2b7435f07778f49656e5a8f",
    "num_train_epochs": 3
  },
  destination="kevinlu625/hfraudtest"
)

print(training)


# llm = Replicate(
#     model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
#     model_kwargs={"temperature": 0.75, "max_length": 3000}
# )

# qa_chain = ConversationalRetrievalChain.from_llm(
#     llm,
#     vectordb.as_retriever(search_kwargs={'k': 2}),
#     return_source_documents=True
# )

# chat_history = []
# while True:
#     query = input('Prompt: ')
#     result = qa_chain({'question': query, 'chat_history': chat_history})
#     print('Answer: ' + result['answer'] + '\n')
#     chat_history.append((query, result['answer']))