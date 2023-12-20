import os
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader

os.environ['REPLICATE_API_TOKEN'] = "r8_TPXMKMNaTtXLNn4grSg5qibyIQVDJw81AixsC"

loader = CSVLoader(file_path="/Users/kevinlu/Desktop/claims data/Train_Inpatientdata-1542865627584.csv")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

pinecone.init(      
	api_key='4fa75472-35d7-4ae5-b9aa-de9bfb0f8a5d',      
	environment='gcp-starter'      
)      
index = pinecone.Index('hfraudtest')
vectordb = Pinecone.from_existing_index('hfraudtest', embeddings)

llm = Replicate(
    model="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
    model_kwargs={"temperature": 0.75, "max_length": 3000}
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True
)

chat_history = []
while True:
    query = input('Prompt: ')
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    result = qa_chain({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'] + '\n')
    chat_history.append((query, result['answer']))