import os
import pinecone
import environ
import time
import pandas as pd
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone 
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DataFrameLoader
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from tqdm.auto import tqdm

env = environ.Env()
environ.Env.read_env()

REPLICATE_API_TOKEN = env('REPLICATE_API_TOKEN')
PINECONE_API_KEY = env("PINECONE_API_KEY")
OPENAI_API_KEY = env('OPENAI_API_KEY')

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

dojPressRelease = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/data/dojPressRelease.csv')
def runDOJPressReleaseRagWithGPT3Point5Turbo(dojPressRelease):
    #testing out RAG deployment
    chat = ChatOpenAI(
        openai_api_key=OPENAI_API_KEY,
        model='gpt-3.5-turbo'
    )

    pinecone.init(      
        api_key=PINECONE_API_KEY,      
        environment='gcp-starter'      
    )      

    index_name = 'hfraudcontext'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )
        # wait for index to finish initialization
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pinecone.Index(index_name)

    # print(index.describe_index_stats())

    def chunkingDataDownIntoParagraphs(dojPressRelease):
        # Splitting paragraphs
        dojPressRelease['body'] = dojPressRelease['body'].str.split('\n\n')

        # Exploding the DataFrame to create separate rows for each paragraph
        dojPressRelease = dojPressRelease.explode('body')

        # Resetting index
        dojPressRelease.reset_index(drop=True, inplace=True)

        dojPressRelease = dojPressRelease.loc[:, ~dojPressRelease.columns.str.contains('^Unnamed')]

        dojPressRelease['id'] = dojPressRelease.reset_index().index

        dojPressRelease['title'] = dojPressRelease['title'].astype(str)
        dojPressRelease['body'] = dojPressRelease['body'].astype(str)
        dojPressRelease['date'] = dojPressRelease['date'].astype(str)

    embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    def upsertingVectorsToPinecone(dojPressReleaseCSV):
        batch_size = 100

        for i in tqdm(range(0, len(dojPressReleaseCSV), batch_size)):
            i_end = min(len(dojPressReleaseCSV), i + batch_size)
            
            # Get batch of data
            batch = dojPressReleaseCSV.iloc[i:i_end]
            
            # Get text to embed
            texts = [x['body'] for _, x in batch.iterrows()]

            ids = [f"{x['id']}-{i}" for i, x in batch.iterrows()]
            
            # Embed text
            embeds = embed_model.embed_documents(texts)
            
            # Get metadata to store in Pinecone
            metadata = [
                {'text': x['body'],
                 'date': x['date'],
                 'title': x['title']} for _, x in batch.iterrows()
        ]
        
        # Add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))

    text_field = "text"  # the metadata field that contains our text

    # initialize the vector store object
    vectorstore = Pinecone(
        index, embed_model.embed_query, text_field
    )

    query = "Give me an example of a fraudulent healthcare scheme."

    # print(vectorstore.similarity_search(query, k=5))

    def augment_prompt(query: str):
        # get top 3 results from knowledge base
        results = vectorstore.similarity_search(query, k=5)
        # get the text from the results
        source_knowledge = "\n".join([x.page_content for x in results])
        # feed into an augmented prompt
        augmented_prompt = f"""Using the contexts below, answer the query.

        Contexts:
        {source_knowledge}

        Query: {query}"""

        print(results)

        return augmented_prompt

    prompt_text = input("Describe your case: ")

    prompt = HumanMessage(
        content=augment_prompt(
            prompt_text
        )
    )

    messages = [
        SystemMessage(content="You are an LLM trained on historic healthcare fraud cases."),
        HumanMessage(content="Hi AI, how are you today?"),
        AIMessage(content="I'm great thank you. How can I help you?"),
    ]

    res = chat(messages + [prompt])

    messages.append(res)

    print(res.content)






######################################

# loader = DataFrameLoader(resultingInpatientStrgData, page_content_column="sentence")
# documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = HuggingFaceEmbeddings()

# pinecone.init(      
# 	api_key=PINECONE_API_KEY,      
# 	environment='gcp-starter'      
# )      
# index_name = "hfraud"
# index = pinecone.Index(index_name)

# #adding to Pinecone index
# # vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# #using Pinecone index
# vectordb = Pinecone.from_existing_index("stringeddata", embeddings)

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