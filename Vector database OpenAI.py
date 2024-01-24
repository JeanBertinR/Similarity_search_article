# Import the packages
import os
import openai
from dotenv import load_dotenv
import pandas as pd



# Set openai.api_key to the OPENAI environment variable
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base  = os.environ["OPENAI_API_BASE"]
openai.api_version  = os.environ["OPENAI_API_VERSION"]

# Load data from the parquet file
dataframe = pd.read_parquet('data/default_train_0000.parquet', engine='fastparquet')[["sentence_A", "sentence_B", "relatedness_score"]]

# Create a new column called "model_result"
dataframe["model_result"] = None

# Define the system message
system_msg = 'You are a helpful assistant who help to retrieve similarity score between two sentences.'

# Iterate over the DataFrame using a for loop with an iteration of the type for row_number in range(number_of_lines)
#for row_number in range(len(dataframe)):
for row_number in range(10):

    # Get the row at the current index
    row = dataframe.loc[row_number]
    print("Processing: ",row_number)

    # Define the user message
    user_msg = 'Pleae give me the similarity score from 0 to 1 between those sentences : "' + row["sentence_A"] +  '" and "' + row["sentence_B"] + '". Always respond using stricly and only the following format : Similarity_score : XXX "'
    print(user_msg)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", deployment_id="s4u-dev-open_ai_deployment_chat",messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}])

    # Retrieve similarity score
    final_score = response["choices"][0]["message"]["content"].replace("Similarity_score :", "")

    # Store the similarity score in the DataFrame
    dataframe.loc[row_number, "model_result"] = final_score



# Create a dataset using GPT
#response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
#response = openai.ChatCompletion.create(model="gpt-4",deployment_id = "chat4",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "s4u-dev-open_ai_deployment_chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])

