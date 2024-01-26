# Import the packages
import os
import openai
from dotenv import load_dotenv
import pandas as pd
from scipy.stats import pearsonr


# Set openai.api_key to the OPENAI environment variable
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base  = os.environ["OPENAI_API_BASE"]
openai.api_version  = os.environ["OPENAI_API_VERSION"]

# Load data from the parquet files
df_train = pd.read_parquet('data/biosses_train_0000.parquet', engine='fastparquet')
df_test = pd.read_parquet('data/biosses_test_0000.parquet', engine='fastparquet')

# Create a new column called "model_result"
df_test["model_score"] = None

# Build the prompt with examples from the train file
train_examples = ""
#for row_number in range(len(df_train)):
for row_number in range(24):

    # Get the row at the current index
    row = df_train.loc[row_number]
    chain = 'The sentence "' + row["sentence1"] +  '" and the sentence "' + row["sentence1"] + '" have a similarity score of  ' + row["score"].astype(str)
    train_examples += chain + "\n"


system_msg = '''You are a helpful assistant who helps retrieve similarity scores between two sentences.
You will find below some examples to help you determine this similarity score with the best accuracy :
''' + train_examples


print(system_msg)

# Iterate over the DataFrame using a for loop with an iteration of the type for row_number in range(number_of_lines)
for row_number in range(len(df_test)):

    # Get the row at the current index
    row = df_test.loc[row_number]
    print("Processing: ",row_number)

    # Define the user message
    user_msg = 'Please give me the similarity score from 0 to 4 between those sentences : "' + row["sentence1"] +  '" and "' + \
               row["sentence2"] + \
               '". Always respond using strictly and only the following format : Similarity_score : XXX "'

    print(user_msg)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            deployment_id="s4u-dev-open_ai_deployment_chat",
                                            temperature = 0,
                                            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}])

    try:
        # Retrieve similarity score
        final_score = response["choices"][0]["message"]["content"].split(":")[1].strip()

        # Store the similarity score in the DataFrame
        df_test.loc[row_number, "model_score"] = final_score

    except Exception as e:
        print(f"An unexpected error occurred : {e}")

# Convertir la colonne 'model_score' en type float
df_test['model_score'] = df_test['model_score'].astype(float)

# Calculate Person correlation between score and model_score
correlation_coefficient, _ = pearsonr(df_test['score'], df_test['model_score'])



# Create a dataset using GPT
#response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
#response = openai.ChatCompletion.create(model="gpt-4",deployment_id = "chat4",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
#response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "s4u-dev-open_ai_deployment_chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])

