# Import the packages
import os
import openai
from dotenv import load_dotenv

load_dotenv()

# Set openai.api_key to the OPENAI environment variable
openai.api_key = os.environ["OPENAI"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base  = os.environ["OPENAI_API_BASE"]
openai.api_version  = os.environ["OPENAI_API_VERSION"]

# Define the system message
system_msg = 'You are a helpful assistant who understands data science.'

# Define the user message
user_msg = 'Pleae give me the similarity score from 0 to 1 between those sentences "I need to go home" and "I want to go home. Always respond using the following format : Similarity_score : XXX "'

# Create a dataset using GPT
#response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
#response = openai.ChatCompletion.create(model="gpt-4",deployment_id = "chat4",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
response = openai.ChatCompletion.create(model="gpt-35-turbo",deployment_id = "s4u-dev-open_ai_deployment_chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])

response["choices"][0]["message"]["content"]