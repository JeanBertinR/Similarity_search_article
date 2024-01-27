# Import the packages
import os
import openai
from dotenv import load_dotenv
import pandas as pd
from scipy.stats import pearsonr
import re

# Set up openai environment variables
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
openai.api_type = os.environ["OPENAI_API_TYPE"]
openai.api_base  = os.environ["OPENAI_API_BASE"]
openai.api_version  = os.environ["OPENAI_API_VERSION"]
model_deployment = os.environ["DEPLOYMENT_ID"]

# Load data from the parquet files
df_train = pd.read_parquet('data/biosses_train_0000.parquet', engine='fastparquet')
df_test = pd.read_parquet('data/biosses_test_0000.parquet', engine='fastparquet')


# Define correlation function
def calculate_correlation(example_size,temperature):

    # Create a new column called "model_result"
    df_test["model_score"] = None

    # Build the prompt with examples from the train file
    train_examples = ""
    #for row_number in range(len(df_train)):
    for row_number in range(example_size):

        # Get the row at the current index
        row = df_train.loc[row_number]
        chain = 'The sentence "' + row["sentence1"] +  '" and the sentence "' + row["sentence1"] + '" have a similarity score of  ' + row["score"].astype(str)
        train_examples += chain + "\n"


    system_msg = '''You are a helpful assistant who helps retrieve similarity scores between two sentences.
    You will find below some examples to help you determine this similarity score with the best accuracy :
    ''' + train_examples


    # Iterate over the DataFrame using a for loop with an iteration of the type for row_number in range(number_of_lines)
    for row_number in range(len(df_test)):

        # Get the row at the current index
        row = df_test.loc[row_number]

        # Define the user message
        user_msg = 'Please give me the similarity score from 0.0 to 4.0 between those sentences : "' + row["sentence1"] +  '" and "' + \
                row["sentence2"] + \
                '". Always respond using strictly and only the following format without any justification: Similarity_score : XXX "'

        response = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            deployment_id=model_deployment,
                                            temperature = temperature,
                                            seed = 42,
                                            messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}])


        try:
            # Retrieve similarity score
            final_score = response["choices"][0]["message"]["content"].split(":")[1].strip()
            final_score = re.sub(r'[^\d.]+$', '', final_score).rstrip('.')
            final_score = float(final_score)

            # Store the similarity score in the DataFrame
            df_test.loc[row_number, "model_score"] = final_score

        except Exception as e:
            print(response)
            print(f"An unexpected error occurred for row_number {row_number}")

    # Convert 'model_score' column to float
    #df_test['model_score'] = df_test['model_score'].astype(float)
    print(df_test)

    # Calculate Person correlation between score and model_score
    correlation_coefficient, _ = pearsonr(df_test['score'], df_test['model_score'])

    return correlation_coefficient, df_test


# Initialize empty list
correlation_values = []

# Iterate on the number of examples given to the prompt
for example in range(0, 70, 10):
    print("Number of example considered in prompt : ",example)
    row_values = []  # Initialiser une nouvelle ligne pour chaque valeur de k
    for temperature in range(11):
        print("Temperature : ", temperature/10)
        result = calculate_correlation(example, temperature/10)[0]
        row_values.append(result)

    # Ajouter la ligne complète au tableau principal
    correlation_values.append(row_values)

# Create dataframe from list
df_correlation = pd.DataFrame(correlation_values, columns=[f"temp_{temperature/10}" for temperature in range(11)], index=[f"example_{example}" for example in range(0, 70, 10)])




calculate_correlation(50,0.0)[1]




# Create a dataset using GPT
#response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
#response = openai.ChatCompletion.create(model="gpt-4",deployment_id = "chat4",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])
#response = openai.ChatCompletion.create(model="gpt-3.5-turbo",deployment_id = "s4u-dev-open_ai_deployment_chat",messages=[{"role": "system", "content": system_msg},{"role": "user", "content": user_msg}])

