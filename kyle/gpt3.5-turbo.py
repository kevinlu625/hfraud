import openai

# export OPENAI_API_KEY='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'

# read openai key from openai-key.txt
with open('kyle/openai-key.txt', 'r') as f:
    key = f.read()

openai.api_key = key
client = openai.OpenAI()

inpatient_data_path = 'inpatientJSONGPT.jsonl'

# client.files.create(
#   file=open(inpatient_data_path, "rb"),
#   purpose="fine-tune"
# )

# client.fine_tuning.jobs.create(
#   training_file="inpatientJSONGPT", 
#   model="gpt-3.5-turbo"
# )

response = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-1106:personal::8ZrzBZSq",
  messages=[
    {"role": "system", "content": "KSol is a chatbot that detects whether or not a patient claim is fraudulent"},
    {"role": "user", "content": "Give me three example claims that are fraudulent, and what pattern do you notice?"}, 
  ]
)
print(response.choices[0].message)