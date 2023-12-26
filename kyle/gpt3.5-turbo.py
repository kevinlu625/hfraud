import openai

# read openai key from openai-key.txt
with open('kyle/openai-key.txt', 'r') as f:
    key = f.read()

openai.api_key = key
client = openai.OpenAI()

inpatient_data_path = 'inpatientJSONGPT.jsonl'

client.files.create(
  file=open(inpatient_data_path, "rb"),
  purpose="fine-tune"
)

client.fine_tuning.jobs.create(
  training_file="inpatientJSONGPT", 
  model="gpt-3.5-turbo"
)



