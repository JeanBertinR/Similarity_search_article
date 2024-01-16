# Import librairies
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd
import fastparquet as fp

# Load data from the parquet file
dataframe = pd.read_parquet('data/default_train_0000.parquet', engine='fastparquet')[["sentence_A", "sentence_B", "relatedness_score"]]



# Define embedding
embedding_engine = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-base-en",
    model_kwargs = {"device":"cpu"},
    encode_kwargs = {'normalize_embeddings':True}

)

# Charger le modèle pré-entraîné
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Obtenir les embeddings pour les phrases de chaque colonne
embeddings_sentence_A = model.encode(dataframe['sentence_A'].tolist(), convert_to_tensor=True)
embeddings_sentence_B = model.encode(dataframe['sentence_B'].tolist(), convert_to_tensor=True)

# Concaténer les embeddings pour former un seul ensemble d'embeddings
embeddings_combined = torch.cat([embeddings_sentence_A, embeddings_sentence_B], dim=1)

# Calculer la similarité entre les paires d'ensembles d'embeddings
similarity_scores = util.pytorch_cos_sim(embeddings_combined, embeddings_combined)

# Afficher les scores de similarité pour les 5 premières paires de phrases
for i in range(5):
    print(f"Similarity Score for pair {i+1}: {similarity_scores[i+1, 0].item()}")






# Charge environment variables
load_dotenv()

OPENAI_API_TYPE = os.environ["OPENAI_API_TYPE"]
OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_API_VERSION = os.environ["OPENAI_API_VERSION"]

# Prepare data1
data1_df = pd.read_csv("data/Legacy_EP5_SAKNR.csv", encoding="utf-8")
data1_df.fillna("O", inplace=True)

# Prepare data2
data2_df = pd.read_csv("data/PEO_SAKNR_detailed.csv", encoding="utf-8")
data2_df.fillna("O", inplace=True)

# Embedding definition
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", deployment="text-embedding-ada-002", chunk_size=16, request_timeout=100)

# Create an empty DataFrame for the final results
final_dataframe = pd.DataFrame(columns=['TEXT_LEGACY', 'TEXT_PEO','CONFIDENCE'])

# Iteration on data1 element to create each time a specific prompt : the iteration is done on the TEXT_LEGACY elements
for index, row in data1_df.drop_duplicates()[201:300].iterrows():

    text_legacy_value = row['TEXT_LEGACY']
    mitkz_value = row['MITKZ']
    print(mitkz_value)
    xopvw_value = row['XOPVW']
    print(xopvw_value)
    xlglr_value = row['XLGCLR']
    print(xlglr_value)


    data2_filtered = data2_df.loc[(data2_df['MITKZ'] == mitkz_value) & (data2_df['XOPVW'] == xopvw_value) & (data2_df['XLGLR'] == xlglr_value)].drop_duplicates()
    data2_filtered['TEXT_PEO'] = data2_filtered['TEXT_PEO'].apply(lambda x: "'TEXT_PEO' :'" + x)

    loader = DataFrameLoader(data2_filtered, page_content_column = 'TEXT_PEO').load()


    # Rest of your processing code for merged_data
    vectorstore = FAISS.from_documents(loader, embeddings)
    #chain = ConversationalRetrievalChain.from_llm(llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', deployment_id="chat", request_timeout=100), retriever=vectorstore.as_retriever())


    print(f"Processing TEXT_LEGACY: {text_legacy_value}")  # Ajout du message d'impression
    result = vectorstore.similarity_search_with_relevance_scores(text_legacy_value, k=1)
    response = result[0][0].page_content.split("'TEXT_PEO' :'")[1]
    score_confidence = result[0][1]

    # Create a matching table for this iteration
    matching_table = pd.DataFrame({'TEXT_LEGACY': [text_legacy_value], 'TEXT_PEO': [response], 'CONFIDENCE': [score_confidence]})

    # Concatenate the matching table to the final DataFrame
    final_dataframe = pd.concat([final_dataframe, matching_table], ignore_index=True)


# Merge with PEO and LEGACY sources to retrieve the other columns
legacy_merge = pd.merge(final_dataframe, data1_df, on="TEXT_LEGACY", how='inner')
final_matching_table = pd.merge(legacy_merge, data2_df, on="TEXT_PEO", how='inner')
final_matching_table1 = final_matching_table[final_matching_table['XOPVW_x'] == final_matching_table['XOPVW_y']]
final_matching_table2 = final_matching_table1[final_matching_table1['XOPVW_x'] == final_matching_table1['XOPVW_y']]
final_matching_table3 = final_matching_table2[final_matching_table2['XOPVW_x'] == final_matching_table2['XOPVW_y']]



# Export final DataFrame
final_matching_table3.to_excel("data/Matching_result_similarity_search.xlsx", index=False)

