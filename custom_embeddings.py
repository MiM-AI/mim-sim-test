import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import boto3
import json
from io import BytesIO
from openai import OpenAI
from dotenv import load_dotenv

# imports
from typing import List, Tuple  # for type hints

import numpy as np  # for manipulating arrays
import pandas as pd  # for manipulating data in dataframes
import pickle  # for saving the embeddings cache
import plotly.express as px  # for plots
import random  # for generating run IDs
from sklearn.model_selection import train_test_split  # for splitting train & test data
import torch  # for matrix optimization

import openai
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from langchain.embeddings import OpenAIEmbeddings
# Function to compute cosine similarity between two embeddings
from numpy import dot
from numpy.linalg import norm

# from utils.embeddings_utils import get_embedding, cosine_similarity  # for embeddings

from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

embeddings_model = OpenAIEmbeddings()

# Initialize a session using the 'personal' profile
session = boto3.Session(profile_name='personal')
s3 = session.client('s3')  

bucket_name = "mim-course-catalogs-jsons"

def list_files_in_bucket(bucket_name, prefix=''):
    files = []
    result = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # Check if there are contents in the result
    while result.get('Contents'):
        for content in result['Contents']:
            files.append(content['Key'])
        if result['IsTruncated']:  # If there are more files to retrieve
            continuation_token = result['NextContinuationToken']
            result = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            break
    return files

def load_json_from_s3(bucket_name, key):
    obj = s3.get_object(Bucket=bucket_name, Key=key)
    data = json.load(BytesIO(obj['Body'].read()))
    return data

def load_inst_data():
	files = list_files_in_bucket(bucket_name)

	all_data = []
	for file in files:
		print(file)
		inst = file.split("/")[0]
		year = file.split("/")[1].split("-")[0]
		
		json_data = load_json_from_s3(bucket_name, file)
		df = pd.DataFrame(json_data)
		df.insert(0, "INSTITUTION", inst)
		df.insert(1, "YEAR", year)
		
		all_data.append(df)

	retdat = pd.concat(all_data)
	return retdat

external_institution = "Portland Community College (PCC)"
def get_articulation_dat(external_institution):
    adat = pd.read_csv('data/Articulation_data.csv', encoding="ISO-8859-1")
    pdat = adat[adat['TRNS_DESCRIPTION'] == external_institution]

    pdat = pdat[['EQUIV_EXISTS', 'TRNS_DESCRIPTION', 'TR_SUBJ', 'TR_COURSE', 'TR_TITLE', 'TERM', 'EQUIV_SUBJECT', 'EQUIV_CRSE', 'EQUI_TITLE']]
    pdat = pdat[pdat['EQUIV_EXISTS'] == "Y"]

    pdat = pdat[pdat['EQUI_TITLE'] != 'NOT ACCEPTED IN TRANSFER']
    # pdat = pdat[pdat['EQUIV_SUBJECT'] != "0000"]
    # pdat = pdat[pdat['EQUIV_SUBJECT'] != "000"]
    # pdat = pdat[pdat['EQUIV_CRSE'] != "000"]
    # pdat = pdat[pdat['EQUIV_CRSE'] != "0000"]
    # pdat = pdat[pdat['TR_TITLE'] != "000"]
    # pdat = pdat[pdat['TR_TITLE'] != "0000"]

    pdat['EQUIV_CRSE'].unique()
    pdat = pdat[pdat['EQUIV_CRSE'].str.match(r'^\d+$')]
    pdat = pdat[pdat['EQUIV_CRSE'] != "000"]
    pdat = pdat[pdat['EQUIV_CRSE'] != "0000"]

    # pdat = pdat[pdat['EQUIV_CRSE'].isin(["LDT", "UDT", "LDU", "UDU", "UST", "LTD", "LDTZ"])]

    pdat = pdat[(pdat['TERM'] >= 200700)]
    pdat['YEAR'] = (np.floor(pdat['TERM'] / 100).astype(int).astype(str))
    pdat['EXT COURSE CODE'] = pdat['TR_SUBJ'] + " " + pdat['TR_COURSE']
    pdat['EXT TITLE'] = pdat['TR_TITLE']

    pdat['INT COURSE CODE'] = pdat['EQUIV_SUBJECT'] + " " + pdat['EQUIV_CRSE']
    pdat['INT TITLE'] = pdat['EQUI_TITLE']

    pdat['EQUIV_SUBJECT'].unique()

    pdat = pdat[['YEAR', 'TRNS_DESCRIPTION', 'EXT COURSE CODE', 'EXT TITLE', 'INT COURSE CODE', 'INT TITLE']].reset_index(drop=True)
    pdat = pdat.drop_duplicates(subset=['YEAR', 'EXT COURSE CODE'])

    return pdat

def load_osu():
	odat = pd.read_excel('data/CIM approved snapshot 2024-07-15.xlsx')
	odat.columns

	odat = odat[['Code', 'Description (50 to 80 words is ideal) (description)']]
	odat.columns = ['INT COURSE CODE', 'INT DESCRIPTION']
	return odat



# Get OSU data
odat = load_osu()

# Get institution data
inst_dat = load_inst_data()
idat = inst_dat[['INSTITUTION', 'YEAR', 'COURSE CODE', 'DESCRIPTION']]

adat = pd.read_csv('data/Articulation_data.csv', encoding="ISO-8859-1")
adat[adat['TRNS_DESCRIPTION'].str.contains('Western Oregon')]

college_list = ['Central Oregon Com College', 'Chemeketa Community College', 'Clackamas Community College',
'Lane Community College', 'Linn-Benton Community College', 'Mt Hood Community College',
'Oregon Inst of Technology', 'Portland Community College (PCC)', 'University of Oregon',
'Western Oregon University']


rename_dict = {'Central Oregon Com College': 'Central-Oregon-Community-College', 
  'Chemeketa Community College': 'Chemeketa-Community-College', 
  'Clackamas Community College': 'Clackamas-Community-College',
  'Lane Community College': 'Lane-Community-College',
  'Linn-Benton Community College': 'Linn-Benton-Community-College', 
  'Mt Hood Community College': 'Mt.-Hood-Community-College',
  'Oregon Inst of Technology': 'Oregon Inst of Technology',
  'Portland Community College (PCC)': 'Portland-Community-College', 
  'University of Oregon': 'University-of-Oregon',
  'Western Oregon University': 'Western-Oregon-University'
}


dat = [get_articulation_dat(x) for x in college_list]
dat = pd.concat(dat)
dat['TRNS_DESCRIPTION'] = dat['TRNS_DESCRIPTION'].map(rename_dict).fillna(dat['TRNS_DESCRIPTION'])

mdat = dat.merge(idat, left_on=['TRNS_DESCRIPTION', 'YEAR', 'EXT COURSE CODE'], right_on=['INSTITUTION', 'YEAR', 'COURSE CODE'], how='left')
mdat = mdat.merge(odat, on=['INT COURSE CODE'], how='left')
mdat = mdat.dropna(subset=['DESCRIPTION', 'INT DESCRIPTION'])
mdat

df = mdat[['DESCRIPTION', 'INT DESCRIPTION']].reset_index(drop=True)
df['label'] = 1
df.columns = ['text_1', 'text_2', 'label']



# ---------------------------------------------
# Custom Embededings

# Path to cache file
embedding_cache_path = "embeddings/custom_embeddings_model/custom_embedding_cache.pkl"

# Initialize an empty cache or load existing one
if os.path.exists(embedding_cache_path):
    with open(embedding_cache_path, "rb") as f:
        embedding_cache = pickle.load(f)
else:
    embedding_cache = {}

def dataframe_of_negatives(dataframe_of_positives: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe of negative pairs made by combining elements of positive pairs."""
    texts = set(dataframe_of_positives["text_1"].values) | set(
        dataframe_of_positives["text_2"].values
    )
    all_pairs = {(t1, t2) for t1 in texts for t2 in texts if t1 < t2}
    positive_pairs = set(
        tuple(text_pair)
        for text_pair in dataframe_of_positives[["text_1", "text_2"]].values
    )
    negative_pairs = all_pairs - positive_pairs
    df_of_negatives = pd.DataFrame(list(negative_pairs), columns=["text_1", "text_2"])
    df_of_negatives["label"] = -1
    return df_of_negatives


# Function to get embedding from OpenAI
def get_embedding(text: str, model: str = "text-embedding-ada-002") -> list:
    # Using the openai.Embedding.create() method
    response = openai.Embedding.create(input=text, model=model)
    return response.data[0].embedding  # Updated way to access the embedding

# Function to retrieve embeddings from cache, or compute and save them
def get_embedding_with_cache(
    text: str,
    embedding_cache: dict = embedding_cache,
    embedding_cache_path: str = embedding_cache_path,
) -> list:
    if text not in embedding_cache:
        # If the embedding is not in the cache, compute it using LangChain's embed_documents
        print(f"Generating Embeddings for: {text}")
        embedding_cache[text] = embeddings_model.embed_documents([text])[0]
        # Save the updated cache to disk
        with open(embedding_cache_path, "wb") as f:
            pickle.dump(embedding_cache, f)
    return embedding_cache[text]


def cosine_similarity(embedding1, embedding2):
    return dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))


def accuracy_and_se(cosine_similarity: float, labeled_similarity: int) -> Tuple[float]:
    accuracies = []
    for threshold_thousandths in range(-1000, 1000, 1):
        threshold = threshold_thousandths / 1000
        total = 0
        correct = 0
        for cs, ls in zip(cosine_similarity, labeled_similarity):
            total += 1
            if cs > threshold:
                prediction = 1
            else:
                prediction = -1
            if prediction == ls:
                correct += 1
        accuracy = correct / total
        accuracies.append(accuracy)
    a = max(accuracies)
    n = len(cosine_similarity)
    standard_error = (a * (1 - a) / n) ** 0.5  # standard error of binomial
    return a, standard_error


def embedding_multiplied_by_matrix(
    embedding: List[float], matrix: torch.tensor
) -> np.array:
    embedding_tensor = torch.tensor(embedding).float()
    modified_embedding = embedding_tensor @ matrix
    modified_embedding = modified_embedding.detach().numpy()
    return modified_embedding

# compute custom embeddings and new cosine similarities
def apply_matrix_to_embeddings_dataframe(matrix: torch.tensor, df: pd.DataFrame):
    for column in ["text_1_embedding", "text_2_embedding"]:
        df[f"{column}_custom"] = df[column].apply(
            lambda x: embedding_multiplied_by_matrix(x, matrix)
        )
    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(
            row["text_1_embedding_custom"], row["text_2_embedding_custom"]
        ),
        axis=1,
    )


def optimize_matrix(
    modified_embedding_length: int = 2048,  # in my brief experimentation, bigger was better (2048 is length of babbage encoding)
    batch_size: int = 100,
    max_epochs: int = 100,
    learning_rate: float = 100.0,  # seemed to work best when similar to batch size - feel free to try a range of values
    dropout_fraction: float = 0.0,  # in my testing, dropout helped by a couple percentage points (definitely not necessary)
    df: pd.DataFrame = df,
    print_progress: bool = True,
    save_results: bool = True,
) -> torch.tensor:
    """Return matrix optimized to minimize loss on training data."""
    run_id = random.randint(0, 2 ** 31 - 1)  # (range is arbitrary)
    # convert from dataframe to torch tensors
    # e is for embedding, s for similarity label
    def tensors_from_dataframe(
        df: pd.DataFrame,
        embedding_column_1: str,
        embedding_column_2: str,
        similarity_label_column: str,
    ) -> Tuple[torch.tensor]:
        e1 = np.stack(np.array(df[embedding_column_1].values))
        e2 = np.stack(np.array(df[embedding_column_2].values))
        s = np.stack(np.array(df[similarity_label_column].astype("float").values))

        e1 = torch.from_numpy(e1).float()
        e2 = torch.from_numpy(e2).float()
        s = torch.from_numpy(s).float()

        return e1, e2, s

    e1_train, e2_train, s_train = tensors_from_dataframe(
        df[df["dataset"] == "train"], "text_1_embedding", "text_2_embedding", "label"
    )
    e1_test, e2_test, s_test = tensors_from_dataframe(
        df[df["dataset"] == "test"], "text_1_embedding", "text_2_embedding", "label"
    )

    # create dataset and loader
    dataset = torch.utils.data.TensorDataset(e1_train, e2_train, s_train)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # define model (similarity of projected embeddings)
    def model(embedding_1, embedding_2, matrix, dropout_fraction=dropout_fraction):
        e1 = torch.nn.functional.dropout(embedding_1, p=dropout_fraction)
        e2 = torch.nn.functional.dropout(embedding_2, p=dropout_fraction)
        modified_embedding_1 = e1 @ matrix  # @ is matrix multiplication
        modified_embedding_2 = e2 @ matrix
        similarity = torch.nn.functional.cosine_similarity(
            modified_embedding_1, modified_embedding_2
        )
        return similarity

    # define loss function to minimize
    def mse_loss(predictions, targets):
        difference = predictions - targets
        return torch.sum(difference * difference) / difference.numel()

    # initialize projection matrix
    embedding_length = len(df["text_1_embedding"].values[0])
    matrix = torch.randn(
        embedding_length, modified_embedding_length, requires_grad=True
    )

    epochs, types, losses, accuracies, matrices = [], [], [], [], []
    for epoch in range(1, 1 + max_epochs):
        # iterate through training dataloader
        for a, b, actual_similarity in train_loader:
            # generate prediction
            predicted_similarity = model(a, b, matrix)
            # get loss and perform backpropagation
            loss = mse_loss(predicted_similarity, actual_similarity)
            loss.backward()
            # update the weights
            with torch.no_grad():
                matrix -= matrix.grad * learning_rate
                # set gradients to zero
                matrix.grad.zero_()
        # calculate test loss
        test_predictions = model(e1_test, e2_test, matrix)
        test_loss = mse_loss(test_predictions, s_test)

        # compute custom embeddings and new cosine similarities
        apply_matrix_to_embeddings_dataframe(matrix, df)

        # calculate test accuracy
        for dataset in ["train", "test"]:
            data = df[df["dataset"] == dataset]
            a, se = accuracy_and_se(data["cosine_similarity_custom"], data["label"])

            # record results of each epoch
            epochs.append(epoch)
            types.append(dataset)
            losses.append(loss.item() if dataset == "train" else test_loss.item())
            accuracies.append(a)
            matrices.append(matrix.detach().numpy())

            # optionally print accuracies
            if print_progress is True:
                print(
                    f"Epoch {epoch}/{max_epochs}: {dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}"
                )

    data = pd.DataFrame(
        {"epoch": epochs, "type": types, "loss": losses, "accuracy": accuracies}
    )
    data["run_id"] = run_id
    data["modified_embedding_length"] = modified_embedding_length
    data["batch_size"] = batch_size
    data["max_epochs"] = max_epochs
    data["learning_rate"] = learning_rate
    data["dropout_fraction"] = dropout_fraction
    data[
        "matrix"
    ] = matrices  # saving every single matrix can get big; feel free to delete/change
    if save_results is True:
        data.to_csv(f"{run_id}_optimization_results.csv", index=False)

    return data



# -------------------------------------------------
# Start of code to process custom emb

# split data into train and test sets
test_fraction = 0.2  
random_seed = 123  

train_df, test_df = train_test_split(
    df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
)

train_df.loc[:, "dataset"] = "train"
test_df.loc[:, "dataset"] = "test"

negatives_per_positive = (
    1
)

# generate negatives for training dataset
train_df_negatives = dataframe_of_negatives(train_df)
train_df_negatives["dataset"] = "train"

# generate negatives for test dataset
test_df_negatives = dataframe_of_negatives(test_df)
test_df_negatives["dataset"] = "test"

# sample negatives and combine with positives
train_df = pd.concat(
    [
        train_df,
        train_df_negatives.sample(
            n=len(train_df) * negatives_per_positive, random_state=random_seed
        ),
    ]
)

test_df = pd.concat(
    [
        test_df,
        test_df_negatives.sample(
            n=len(test_df) * negatives_per_positive, random_state=random_seed
        ),
    ]
)

df = pd.concat([train_df, test_df])

## Create a column of embeddings for text_1 and text_2
for column in ["text_1", "text_2"]:
    df[f"{column}_embedding"] = df[column].apply(get_embedding_with_cache)

# Create a column for cosine similarity between the two embeddings
df["cosine_similarity"] = df.apply(
    lambda row: cosine_similarity(row["text_1_embedding"], row["text_2_embedding"]),
    axis=1
)

print(df)

for dataset in ["train", "test"]:
    data = df[df["dataset"] == dataset]
    a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
    print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")


results = []
max_epochs = 30
dropout_fraction = 0.2
for batch_size, learning_rate in [(10, 10), (100, 100), (1000, 1000)]:
    result = optimize_matrix(
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        dropout_fraction=dropout_fraction,
        save_results=False,
    )
    results.append(result)


runs_df = pd.concat(results)

# apply result of best run to original data
best_run = runs_df.sort_values(by="accuracy", ascending=False).iloc[0]
best_matrix = best_run["matrix"]

with open("embeddings/custom_embeddings_model/best_matrix.pkl", "wb") as f:
    pickle.dump(best_matrix, f)


# apply_matrix_to_embeddings_dataframe(best_matrix, df)
