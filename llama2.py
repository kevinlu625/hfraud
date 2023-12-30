import os
import environ
import replicate

env = environ.Env()
environ.Env.read_env()

REPLICATE_API_TOKEN = env('REPLICATE_API_TOKEN')

os.environ['REPLICATE_API_TOKEN'] = REPLICATE_API_TOKEN

# llama2 base model training
training = replicate.trainings.create(
  version="meta/llama-2-13b:078d7a002387bd96d93b0302a4c03b3f15824b63104034bfa943c63a8f208c38",
  input={
    "train_data": "https://gist.githubusercontent.com/kevinlu625/8002f65c3bbda92e4fa53eb42feaad5f/raw/763171dc6466c2978a2f19a0ffb073cb5940a29a/inpatientJSON.jsonl",
    "num_train_epochs": 3
  },
  destination="kevinlu625/hfraudtest"
)

print(training)