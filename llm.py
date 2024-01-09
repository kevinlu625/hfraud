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
from openai import OpenAI

env = environ.Env()
environ.Env.read_env()

REPLICATE_API_TOKEN = env('REPLICATE_API_TOKEN')
PINECONE_API_KEY = env("PINECONE_API_KEY")
OPENAI_API_KEY = env('OPENAI_API_KEY')

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# dojPressRelease = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/data/dojPressRelease.csv')
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

client = OpenAI()

response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:k-solutions::8eenabp7",
  messages=[
    {"role": "system", "content": "You are a chatbot that, given some provider data, determines whether or not the provider is fraudulent"},
    {"role": "user", "content": "Please determine whether or not this provider is fraudulent: The provider\u2019s name is Shouping nan Li MD. They practice Emergency Medicine. In 2019, they charged Medicare for these HCPCS codes: ['A7003', 'E0570', 'J7626', 'J7606', 'E0431', 'A7038', 'A4256', 'A7037', 'A7032', 'A4258', 'Q0513', 'A4233', 'E0443', 'E1392', 'E0466', 'J7620', 'J7613', 'E1390', 'A4259', 'A4253']. Their beneficiaries' average age was 75.873015873. Their beneficiaries' average risk score was 1.6290572727. Of this providers patients,  40.0% have Chronic Kidney Disease, 48.0% have Chronic Obstructive Pulmonary Disease, 56.99999999999999% have Diabetes, 21.0% have Depression, 51.0% have Hyperlipidemia, 75.0% have Hypertension, 38.0% have Ischemic Heart Disease, 44.0% have Rheumatoid Arthritis / Osteoarthritis, 0.0% have Schizophrenia, They ordered a total of $284347.34 for all durable medical equipment (DME) products/services. Their Medicare allowed amount for all DME products/services was $77319.77. The amount that Medicare paid them after deductible and coinsurance amounts was $57525.47. The total number of unique beneficiaries associated with DME claims ordered by this provider was 61.0. The total number of DME claims ordered by this provider was 451.0. The total number of unique DME HCPCS codes ordered by this provider was 34.0. The total number of DME products/services ordered by this provider was 9290.0. The total number of DME suppliers this provider worked with was 21.0."}
  ]
)

print(response.choices[0].message)
