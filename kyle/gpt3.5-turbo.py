import openai

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
    {"role": "user", "content": "PRV55907 has a medical claim with id CLM75778 from beneficiary BENE53771 that was started on 2009-11-13 and ended on 2009-11-17 for a reimbursement amount of 5000. The attending physician for this claim was PHY346515 and the operating physician was nan. The claim diagnosis codes were 6826, 25080, 2724, V4364, V090, 2809, 2662, 2720, 27800, nan. The procedure codes were nan, nan, nan.  Is this provider fraudulent?"}
  ]
)
print(completion.choices[0].message)
