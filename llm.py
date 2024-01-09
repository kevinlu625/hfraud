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
    {"role": "user", "content": "Let's find out if this doctor, Jeffrey L Fraser, is doing things the right way or not. In the year 2021, he asked Medicare (which helps people pay for their medical things) for money using a special code called 'A4253'. The people who usually see this doctor are around 68 years old, and the doctor's average score for how risky the patients are is 1.29. Now, this doctor wanted to buy special medical things (like equipment) for a total cost of $924.92. But, Medicare only said it's okay to spend $160.97 on these things. After people paid their part, Medicare gave the doctor $95.88. The doctor asked for this money 12 times and used 5 different codes for these things. In total, the doctor ordered 19 different medical things from 5 different places. Now, when it comes to getting medicine or special food, the doctor didn't ask for any money. The doctor also didn't ask for money for special things like artificial arms or legs. So, the total for these things was $0.00, and Medicare didn't give any money for them. It's a bit like checking if everything is okay with how the doctor is asking for money and what the doctor is buying for the patients."}
  ]
)

print(response.choices[0].message)
