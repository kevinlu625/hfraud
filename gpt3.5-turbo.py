from openai import OpenAI
client = OpenAI()

#training gpt-3.5-turbo
inpatient_data_path = '../inpatientJSON.jsonl'

client.fine_tuning.jobs.create(
  training_file=inpatient_data_path, 
  model="gpt-3.5-turbo"
)



