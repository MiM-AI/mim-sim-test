import os
import json
import pickle
import random 
import ast
from typing import List, Tuple  

import numpy as np
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split  
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import boto3
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from datasets import Dataset
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from numpy import dot
from numpy.linalg import norm
from rake_nltk import Rake



# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")

KEYWORDS = True

def compute_embeddings(df, text_column, col_name='embedding', batch_size=16):
    embeddings = []
    for i in range(0, len(df), batch_size):
        # Extract a batch of texts
        batch_texts = df[text_column].iloc[i:i + batch_size].tolist()
        # Get embeddings for the batch
        batch_embeddings = embeddings_model.embed_documents(batch_texts)
        embeddings.extend(batch_embeddings)
    df[col_name] = [torch.tensor(embed) if embed is not None else None for embed in embeddings]
    return df


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this line to suppress the warning

# Initialize AWS session
session = boto3.Session(profile_name='personal')
s3 = session.client('s3')

bucket_name = "mim-course-catalogs-jsons"

rake = Rake()

# Extract keywords from the DESCRIPTION column
def extract_keywords(description):
    rake.extract_keywords_from_text(description)
    return " ".join(rake.get_ranked_phrases())


def get_embedding(text):
    return torch.tensor(embeddings_model.embed_query(text))


def compute_embeddings(df, text_column, col_name='embedding'):
    df[col_name] = df[text_column].apply(lambda x: get_embedding(x) if isinstance(x, str) and x else None)
    return df


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

        df[df['DESCRIPTION'] == '4 Credits']

        df.insert(0, "INSTITUTION", inst)
        df.insert(1, "YEAR", year)
        
        all_data.append(df)

    retdat = pd.concat(all_data)
    return retdat


def get_articulation_dat(external_institution):
    adat = pd.read_csv('data/Articulation_data.csv', encoding="ISO-8859-1")
    pdat = adat[adat['TRNS_DESCRIPTION'] == external_institution]

    pdat = pdat[['EQUIV_EXISTS', 'TRNS_DESCRIPTION', 'TR_SUBJ', 'TR_COURSE', 'TR_TITLE', 'TERM', 'EQUIV_SUBJECT', 'EQUIV_CRSE', 'EQUI_TITLE']]
    pdat = pdat[pdat['EQUIV_EXISTS'] == "Y"]

    pdat = pdat[pdat['EQUI_TITLE'] != 'NOT ACCEPTED IN TRANSFER']
    pdat = pdat[pdat['EQUIV_CRSE'].str.match(r'^\d+$')]
    pdat = pdat[pdat['EQUIV_CRSE'] != "000"]
    pdat = pdat[pdat['EQUIV_CRSE'] != "0000"]

    pdat = pdat[(pdat['TERM'] >= 200700)]
    pdat['YEAR'] = (np.floor(pdat['TERM'] / 100).astype(int).astype(str))
    pdat['EXT COURSE CODE'] = pdat['TR_SUBJ'] + " " + pdat['TR_COURSE']
    pdat['EXT TITLE'] = pdat['TR_TITLE']

    pdat['INT COURSE CODE'] = pdat['EQUIV_SUBJECT'] + " " + pdat['EQUIV_CRSE']
    pdat['INT TITLE'] = pdat['EQUI_TITLE']

    pdat = pdat[['YEAR', 'TRNS_DESCRIPTION', 'EXT COURSE CODE', 'EXT TITLE', 'INT COURSE CODE', 'INT TITLE']].reset_index(drop=True)
    pdat = pdat.drop_duplicates(subset=['YEAR', 'EXT COURSE CODE'])

    return pdat


def get_neg_articulation_dat(external_institution):
    adat = pd.read_csv('data/Articulation_data.csv', encoding="ISO-8859-1")
    pdat = adat[adat['TRNS_DESCRIPTION'] == external_institution]

    pdat = pdat[['EQUIV_EXISTS', 'TRNS_DESCRIPTION', 'TR_SUBJ', 'TR_COURSE', 'TR_TITLE', 'TERM', 'EQUIV_SUBJECT', 'EQUIV_CRSE', 'EQUI_TITLE']]
    pdat = pdat[pdat['EQUIV_EXISTS'] == "Y"]

    pdat = pdat[pdat['EQUI_TITLE'] == 'NOT ACCEPTED IN TRANSFER']
    pdat = pdat[pdat['EQUIV_CRSE'].str.match(r'^\d+$')]

    pdat = pdat[(pdat['TERM'] >= 200700)]
    pdat['YEAR'] = (np.floor(pdat['TERM'] / 100).astype(int).astype(str))
    pdat['EXT COURSE CODE'] = pdat['TR_SUBJ'] + " " + pdat['TR_COURSE']
    pdat['EXT TITLE'] = pdat['TR_TITLE']

    pdat['INT COURSE CODE'] = pdat['EQUIV_SUBJECT'] + " " + pdat['EQUIV_CRSE']
    pdat['INT TITLE'] = pdat['EQUI_TITLE']

    pdat = pdat[['YEAR', 'TRNS_DESCRIPTION', 'EXT COURSE CODE', 'EXT TITLE', 'INT COURSE CODE', 'INT TITLE']].reset_index(drop=True)
    pdat = pdat.drop_duplicates(subset=['YEAR', 'EXT COURSE CODE'])
    return pdat


def load_osu():
    odat = pd.read_excel('data/CIM approved snapshot 2024-07-15.xlsx')
    odat = odat[['Code', 'Catalog Title (long_title)', 'Description (50 to 80 words is ideal) (description)']]
    odat.columns = ['INT COURSE CODE', 'INT COURSE TITLE', 'INT DESCRIPTION']
    return odat


# Generate negative examples by finding the most similar non-matching courses
def generate_negative_examples(neg_emb, match_emb):
    negative_examples = []

    # Filter match_emb for non-NaN embeddings and reset index
    filtered_match = match_emb[match_emb['embedding'].notna()].reset_index(drop=True)
    match_embeddings = torch.stack(filtered_match['embedding'].tolist())
    match_texts = filtered_match['CODE TITLE DESC 2'].tolist()

    print("Finding the most similar courses from neg_emb in match_emb...")
    for idx, row in neg_emb.iterrows():
        course_desc_1 = row['CODE TITLE DESC 1']
        embedding1 = row['embedding']

        # Skip if the embedding is None
        if embedding1 is None:
            print(f"Skipping '{course_desc_1}' due to missing embedding.")
            continue

        # Compute cosine similarities
        cosine_scores = util.cos_sim(embedding1.unsqueeze(0), match_embeddings)
        
        # Get the highest cosine similarity score and its index
        max_score, top_index = torch.max(cosine_scores, dim=1)
        top_index = top_index.item()
        max_score = max_score.item()
        
        # Retrieve the most similar course description
        most_similar_desc_2 = match_texts[top_index]

        # Debugging output
        print(f"Course: '{course_desc_1}'")
        print(f"Most similar course: '{most_similar_desc_2}' with cosine similarity score: {max_score:.4f}\n")

        # Add to negative examples
        negative_examples.append([course_desc_1, most_similar_desc_2, -1])

    neg_df = pd.DataFrame(negative_examples, columns=['CODE TITLE DESC 1', 'CODE TITLE DESC 2', 'label'])
    return neg_df


# Utility function for cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Calculate accuracy and standard error
# def accuracy_and_se(cosine_similarities: List[float], labels: List[int]) -> Tuple[float, float]:
#     accuracies = []
#     for threshold in np.linspace(0, 1, 2001):  # Generates thresholds from 0 to 1
#         correct = sum(
#             1 for cs, label in zip(cosine_similarities, labels) if (cs > threshold and label == 1) or (cs <= threshold and label == 0)
#         )
#         accuracy = correct / len(cosine_similarities)
#         accuracies.append(accuracy)
    
#     max_accuracy = max(accuracies)
#     n = len(cosine_similarities)
#     se = (max_accuracy * (1 - max_accuracy) / n) ** 0.5  # Standard error of a binomial distribution
#     return max_accuracy, se


def accuracy_and_se(cosine_similarities: List[float], labels: List[int], threshold: float = 0.5) -> Tuple[float, float]:
    correct = sum(
        1 for cs, label in zip(cosine_similarities, labels) if (cs > threshold and label == 1) or (cs <= threshold and label == 0)
    )
    accuracy = correct / len(cosine_similarities)
    n = len(cosine_similarities)
    se = (accuracy * (1 - accuracy) / n) ** 0.5  # Standard error of a binomial distribution
    return accuracy, se

# Function to apply transformation matrix to an embedding
def apply_matrix(embedding: List[float], matrix: torch.Tensor) -> np.ndarray:
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    transformed_embedding = embedding_tensor @ matrix
    return transformed_embedding.detach().numpy()

# Apply matrix transformation to all embeddings in a DataFrame
def apply_matrix_to_embeddings_dataframe(matrix: torch.Tensor, df: pd.DataFrame):
    for col in ["text_1_embedding", "text_2_embedding"]:
        df[f"{col}_custom"] = df[col].apply(lambda x: apply_matrix(x, matrix))
    
    df["cosine_similarity_custom"] = df.apply(
        lambda row: cosine_similarity(row["text_1_embedding_custom"], row["text_2_embedding_custom"]),
        axis=1
    )






# Get OSU data
odat = load_osu()
odat

# Get institution data
inst_dat = load_inst_data()
idat = inst_dat[['INSTITUTION', 'YEAR', 'COURSE CODE', 'COURSE TITLE', 'DESCRIPTION']]

# Filtering to remove non-descriptions
idat = idat[idat['DESCRIPTION'].str.len() > 47]
idat.sample(50)


# List of colleges
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



# Merge articulation data with institutional data then filter nans
mdat = dat.merge(idat, left_on=['TRNS_DESCRIPTION', 'YEAR', 'EXT COURSE CODE'], right_on=['INSTITUTION', 'YEAR', 'COURSE CODE'], how='left')
mdat = mdat.dropna(subset=['INSTITUTION'])





if KEYWORDS == True:
    mdat["DESCRIPTION"] = mdat["DESCRIPTION"].apply(extract_keywords)
    odat['INT DESCRIPTION'] = odat['INT DESCRIPTION'].dropna().apply(extract_keywords)



# Setup COURSE CODE, COURSE TITLE, COURSE DESCRIPTION
mdat['CODE TITLE DESC 1'] = mdat['COURSE CODE'] + ", " + mdat['COURSE TITLE'] + ", " + mdat['DESCRIPTION']
mdat['CODE TITLE DESC 1'] = mdat['CODE TITLE DESC 1'].str.lower()
mdat = mdat.dropna(subset=['CODE TITLE DESC 1'])

odat['CODE TITLE DESC 2'] = odat['INT COURSE CODE'] + ", " + odat['INT COURSE TITLE'] + ", " + odat['INT DESCRIPTION']
odat['CODE TITLE DESC 2'] = odat['CODE TITLE DESC 2'].str.lower()
odat = odat.dropna(subset=['CODE TITLE DESC 2'])

# Merge osu data from current year then filter nans
mdat = mdat.merge(odat, on=['INT COURSE CODE'], how='left')
mdat = mdat.dropna(subset=['INT COURSE CODE', 'DESCRIPTION', 'INT DESCRIPTION'])

df = mdat[['CODE TITLE DESC 1', 'CODE TITLE DESC 2']].reset_index(drop=True)

# Clean up descriptions
# Remove *, + and convert text to lower case
df["CODE TITLE DESC 1"] = df["CODE TITLE DESC 1"].str.replace(r"[*+]", "", regex=True).str.lower()
df["CODE TITLE DESC 2"] = df["CODE TITLE DESC 2"].str.replace(r"[*+]", "", regex=True).str.lower()
df = df.dropna(subset=['CODE TITLE DESC 1', 'CODE TITLE DESC 2'])

df['label'] = 1
df.columns = ['text_1', 'text_2', 'label']


# Get neg course desc not articulated
ndat = [get_neg_articulation_dat(x) for x in college_list]
ndat = pd.concat(ndat)
ndat['TRNS_DESCRIPTION'] = ndat['TRNS_DESCRIPTION'].map(rename_dict).fillna(ndat['TRNS_DESCRIPTION'])

# Merge articulation data with institutional data then filter nans
nmdat = ndat.merge(idat, left_on=['TRNS_DESCRIPTION', 'YEAR', 'EXT COURSE CODE'], right_on=['INSTITUTION', 'YEAR', 'COURSE CODE'], how='left')
nmdat = nmdat.dropna(subset=['INSTITUTION'])



if KEYWORDS == True:
    nmdat["DESCRIPTION"] = nmdat["DESCRIPTION"].apply(extract_keywords)

nmdat['CODE TITLE DESC 1'] = nmdat['COURSE CODE'] + ", " + nmdat['COURSE TITLE'] + ", " + nmdat['DESCRIPTION']
nmdat['CODE TITLE DESC 1'] = nmdat['CODE TITLE DESC 1'].str.lower()




neg_dat = nmdat
match_dat = odat




# Precompute embeddings for neg_dat and match_dat
neg_emb = compute_embeddings(neg_dat, 'CODE TITLE DESC 1')
match_emb = compute_embeddings(match_dat, 'CODE TITLE DESC 2')

# neg_emb.to_pickle("embeddings/neg_emb_large.pkl")
# match_emb.to_pickle("embeddings/match_emb_large.pkl")

neg_emb.to_pickle("embeddings/neg_emb_large_keywords.pkl")
match_emb.to_pickle("embeddings/match_emb_large_keywords.pkl")

# Load the DataFrame with tensors using pickle
# neg_emb = pd.read_pickle("embeddings/neg_emb_large.pkl")
# match_emb = pd.read_pickle("embeddings/match_emb_large.pkl")







# Generate negative examples
neg_df = generate_negative_examples(neg_emb, match_emb)
neg_df.columns = ['text_1', 'text_2', 'label']

# Combine positive and negative examples
mdat = pd.concat([df, neg_df]).reset_index(drop=True)
















# Define or import the accuracy_and_se function
# def accuracy_and_se(predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> Tuple[float, float]:
#     """
#     Calculate accuracy and standard error based on predicted probabilities and binary labels.

#     Args:
#         predictions (np.ndarray): Array of predicted probabilities.
#         labels (np.ndarray): Array of binary labels (0 or 1).
#         threshold (float): Threshold to convert probabilities to binary predictions.

#     Returns:
#         Tuple[float, float]: Accuracy and standard error.
#     """
#     predictions_binary = (predictions >= threshold).astype(int)
#     correct = (predictions_binary == labels).sum()
#     accuracy = correct / len(labels)
#     se = np.sqrt((accuracy * (1 - accuracy)) / len(labels))
#     return accuracy, se

def optimize_matrix(
    modified_embedding_length: int,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    dropout_fraction: float,
    acc_threshold: float,
    df: pd.DataFrame,
    print_progress: bool = True
) -> torch.Tensor:  # Return type updated to return the matrix
    def tensors_from_dataframe(df_subset, embedding_col1, embedding_col2, label_col) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = torch.tensor(np.stack(df_subset[embedding_col1]), dtype=torch.float32)
        e2 = torch.tensor(np.stack(df_subset[embedding_col2]), dtype=torch.float32)
        labels = torch.tensor(df_subset[label_col].values, dtype=torch.float32)
        return e1, e2, labels

    df = df.copy()

    test_fraction = 0.2  
    validation_fraction = 0.1  # Fraction of training data for validation
    random_seed = 123  

    if -1 in df['label'].unique():
        df['label'] = df['label'].apply(lambda x: 0 if x == -1 else 1)


    # Convert labels from -1 and 1 to 0 and 1
    # df['label'] = df['label'].apply(lambda x: 0 if x == -1 else 1)

    # Split the data into training and testing sets
    train_val_df, test_df = train_test_split(
        df, test_size=test_fraction, stratify=df["label"], random_state=random_seed
    )

    # Further split training data into training and validation sets
    train_df, val_df = train_test_split(
        train_val_df, test_size=validation_fraction, stratify=train_val_df["label"], random_state=random_seed
    )

    # Assign dataset labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["dataset"] = "train"
    val_df["dataset"] = "validation"
    test_df["dataset"] = "test"

    # Concatenate the datasets
    dat = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)

    # Extract training, validation, and test data
    e1_train, e2_train, s_train = tensors_from_dataframe(
        dat[dat["dataset"] == "train"], "text_1_embedding", "text_2_embedding", "label"
    )
    e1_val, e2_val, s_val = tensors_from_dataframe(
        dat[dat["dataset"] == "validation"], "text_1_embedding", "text_2_embedding", "label"
    )
    e1_test, e2_test, s_test = tensors_from_dataframe(
        dat[dat["dataset"] == "test"], "text_1_embedding", "text_2_embedding", "label"
    )

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(e1_train, e2_train, s_train),
        batch_size=batch_size,
        shuffle=True
    )

    # Model definition with matrix transformation and dropout
    def model(embedding1, embedding2, matrix, dropout_prob):
        embedding1 = torch.nn.functional.dropout(embedding1, p=dropout_prob, training=True)
        embedding2 = torch.nn.functional.dropout(embedding2, p=dropout_prob, training=True)
        transformed_e1 = embedding1 @ matrix  # Shape: [batch_size, modified_embedding_length]
        transformed_e2 = embedding2 @ matrix  # Shape: [batch_size, modified_embedding_length]
        cosine_sim = torch.nn.functional.cosine_similarity(transformed_e1, transformed_e2)  # Shape: [batch_size]
        return cosine_sim  # Raw scores; sigmoid will be applied in loss

    # **Updated Loss Function: Binary Cross-Entropy with Logits**
    def bce_loss(predictions, targets):
        return torch.nn.functional.binary_cross_entropy_with_logits(predictions, targets)

    embedding_size = len(dat["text_1_embedding"].iloc[0])
    matrix = torch.randn(embedding_size, modified_embedding_length, requires_grad=True)
    # Initialize the matrix using Xavier initialization
    torch.nn.init.xavier_uniform_(matrix)

    # Using Adam optimizer with weight decay for regularization
    optimizer = torch.optim.Adam([matrix], lr=learning_rate, weight_decay=1e-5)
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    results = {"epoch": [], "type": [], "loss": [], "accuracy": [], "auc": []}

    best_val_auc = -np.inf
    best_matrix_state = None

    for epoch in range(1, max_epochs + 1):
        matrix.requires_grad_(True)  # Ensure gradients are tracked for each epoch
        epoch_loss = 0.0  # To accumulate loss over the epoch

        for batch_e1, batch_e2, batch_labels in train_loader:
            optimizer.zero_grad()  # Reset gradients before each step
            predicted_similarity = model(batch_e1, batch_e2, matrix, dropout_fraction)  # Shape: [batch_size]
            loss = bce_loss(predicted_similarity, batch_labels)  # BCEWithLogitsLoss expects raw scores
            loss.backward()
            optimizer.step()  # Update the matrix
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)  # Average loss for the epoch

        # Step the scheduler
        scheduler.step()

        # Evaluate on training, validation, and testing data
        with torch.no_grad():
            matrix.requires_grad_(False)  # Disable gradient tracking during evaluation

            # Function to compute predictions and probabilities
            def get_predictions(e1, e2, s, dataset_label):
                cosine_sim = model(e1, e2, matrix, dropout_fraction).cpu().numpy()
                probabilities = torch.sigmoid(torch.tensor(cosine_sim)).numpy()
                return probabilities

            # Compute predictions
            train_predictions = get_predictions(e1_train, e2_train, s_train, "train")
            val_predictions = get_predictions(e1_val, e2_val, s_val, "validation")
            test_predictions = get_predictions(e1_test, e2_test, s_test, "test")

            # Assign predictions to the DataFrame
            dat.loc[dat["dataset"] == "train", "cosine_similarity_custom"] = train_predictions
            dat.loc[dat["dataset"] == "validation", "cosine_similarity_custom"] = val_predictions
            dat.loc[dat["dataset"] == "test", "cosine_similarity_custom"] = test_predictions

            # Calculate loss on validation set
            val_cosine_sim = model(e1_val, e2_val, matrix, dropout_fraction).cpu()
            val_loss = bce_loss(val_cosine_sim, s_val).item()

            # Calculate loss on test set using BCE Loss
            test_cosine_sim = model(e1_test, e2_test, matrix, dropout_fraction).cpu()
            test_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                test_cosine_sim, s_test
            ).item()

            # Evaluate validation AUC to track the best matrix
            try:
                val_auc = roc_auc_score(s_val.numpy(), val_predictions)
            except ValueError:
                val_auc = float('nan')  # Handle cases where AUC cannot be computed

            # Check if current validation AUC is the best
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_matrix_state = matrix.clone().detach()

            for data_type in ["train", "validation", "test"]:
                subset = dat[dat["dataset"] == data_type]
                similarities = subset["cosine_similarity_custom"].values
                labels = subset["label"].values

                # Calculate metrics
                accuracy, se = accuracy_and_se(similarities, labels, acc_threshold)
                try:
                    auc = roc_auc_score(labels, similarities)
                except ValueError:
                    auc = float('nan')  # Handle cases where AUC cannot be computed

                results["epoch"].append(epoch)
                results["type"].append(data_type)
                if data_type == "train":
                    results["loss"].append(avg_epoch_loss)
                elif data_type == "validation":
                    results["loss"].append(val_loss)
                else:
                    results["loss"].append(test_loss)
                results["accuracy"].append(accuracy)
                results["auc"].append(auc)

                if print_progress:
                    print(f"Epoch {epoch}/{max_epochs}, {data_type}: "
                          f"Loss {results['loss'][-1]:.4f}, "
                          f"Accuracy {accuracy:.1%} ± {1.96 * se:.1%}, "
                          f"AUC {auc:.4f}")

    # After training, load the best matrix state
    if best_matrix_state is not None:
        matrix = best_matrix_state
        print(f"\nBest Validation AUC: {best_val_auc:.4f}")
    else:
        print("\nNo improvement observed on the validation set.")

    return matrix  # Return the optimized matrix




# Assume 'df' is your DataFrame containing 'text_1_embedding' and 'text_2_embedding' columns
# and 'matrix' is the optimized matrix obtained from the 'optimize_matrix' function.

# Function to process the DataFrame and apply matrix transformations
def process_dataframe(matrix: torch.Tensor, df: pd.DataFrame):
    # Apply matrix to embeddings
    apply_matrix_to_embeddings_dataframe(matrix, df)
    
    # Calculate accuracy and standard error for the processed DataFrame
    accuracy, standard_error = accuracy_and_se(df["cosine_similarity_custom"], df["label"])
    
    print(f"Processed DataFrame: Accuracy = {accuracy:.1%} ± {1.96 * standard_error:.1%}")
    
    return df


# Function to compute embeddings for both columns
def add_embeddings_to_dataframe(df):
    # Compute embeddings for text_1
    df = compute_embeddings(df, 'text_1', 'text_1_embedding')
    # Compute embeddings for text_2
    df = compute_embeddings(df, 'text_2', 'text_2_embedding')
    return df

# Assuming `get_ada_embedding` is a function you have defined elsewhere that returns the embedding for a given text
# Now, use this function to add embeddings to your DataFrame
moddat = add_embeddings_to_dataframe(mdat)

moddat.to_pickle("embeddings/model_df_emb_keywords.pkl")

# Load the DataFrame with tensors using pickle
moddat = pd.read_pickle("embeddings/model_df_emb_keywords.pkl")

# Proceed to use the `optimize_matrix` and `process_dataframe` functions after embeddings are added
optimized_matrix = optimize_matrix(
    modified_embedding_length=2048,
    batch_size=32,
    max_epochs=50,
    learning_rate=0.0001,
    dropout_fraction=0.1,
    acc_threshold=0.7,
    df=moddat,
    print_progress=True
)


with open("embeddings/custom_embeddings_model/best_matrix_keywords.pkl", "wb") as f:
    pickle.dump(optimized_matrix, f)



