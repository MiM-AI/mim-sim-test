# imports
import os
import boto3
import json
import pandas as pd
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
import pickle

# For embeddings and model fine-tuning
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from datasets import Dataset

from sklearn.model_selection import train_test_split  # for splitting train & test data
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from typing import List, Tuple  # for type hints
import random  # for generating run IDs
from numpy import dot
from numpy.linalg import norm


# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI()

embeddings_model = OpenAIEmbeddings()

# Initialize AWS session
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


# Central-Oregon-Community-College/2008-2009.json
# Central-Oregon-Community-College/2009-2010.json
# Central-Oregon-Community-College/2010-2011.json
# Central-Oregon-Community-College/2011-2012.json
# Central-Oregon-Community-College/2012-2013.json
# Central-Oregon-Community-College/2013-2014.json
# Central-Oregon-Community-College/2014-2015.json
# Central-Oregon-Community-College/2015-2016.json
# Central-Oregon-Community-College/2016-2017.json
# Central-Oregon-Community-College/2017-2018.json
# file = 'Central-Oregon-Community-College/2018-2019.json'
# Central-Oregon-Community-College/2019-2020.json
# Central-Oregon-Community-College/2020-2021.json
# Central-Oregon-Community-College/2021-2022.json
# Central-Oregon-Community-College/2022-2023.json
# file = 'Central-Oregon-Community-College/2023-2024.json'

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

def load_osu():
    odat = pd.read_excel('data/CIM approved snapshot 2024-07-15.xlsx')
    odat = odat[['Code', 'Description (50 to 80 words is ideal) (description)']]
    odat.columns = ['INT COURSE CODE', 'INT DESCRIPTION']
    return odat

# Get OSU data
odat = load_osu()

# Get institution data
inst_dat = load_inst_data()
idat = inst_dat[['INSTITUTION', 'YEAR', 'COURSE CODE', 'DESCRIPTION']]

# Filtering to remove non-descriptions
idat = idat[idat['DESCRIPTION'].str.len() > 47]
idat.sample(50)

# idat = idat[idat['DESCRIPTION'] != "4 Credits"]
# idat = idat[idat['DESCRIPTION'] != "5 Credits"]
# idat = idat[idat['DESCRIPTION'] != "This course is in development."]
# idat = idat[idat['DESCRIPTION'] != "Terms and hours to be arranged."]



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

mdat = dat.merge(idat, left_on=['TRNS_DESCRIPTION', 'YEAR', 'EXT COURSE CODE'], right_on=['INSTITUTION', 'YEAR', 'COURSE CODE'], how='left')
mdat = mdat.merge(odat, on=['INT COURSE CODE'], how='left')
mdat = mdat.dropna(subset=['DESCRIPTION', 'INT DESCRIPTION'])

df = mdat[['DESCRIPTION', 'INT DESCRIPTION']].reset_index(drop=True)
df['label'] = 1
df.columns = ['text_1', 'text_2', 'label']


# Initialize the pre-trained model
model_name = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model_name)

# Compute embeddings for all unique course descriptions
all_course_descs = pd.concat([df['text_1'], df['text_2']]).unique()
print("Computing embeddings for all unique course descriptions...")
course_desc_embeddings = {}
for desc in all_course_descs:
    if desc not in course_desc_embeddings:
        embedding = model.encode(desc, convert_to_tensor=True)
        course_desc_embeddings[desc] = embedding

# Generate negative examples by finding the most similar non-matching courses
def generate_negative_examples(df):
    negative_examples = []
    # Get unique course_desc_1 and course_desc_2
    course_desc_1_list = df['text_1'].unique()
    course_desc_2_list = df['text_2'].unique()
    
    # Build a set of positive pairs for quick lookup
    positive_pairs = set(zip(df['text_1'], df['text_2']))
    
    print("Generating negative examples by finding most similar non-matching courses...")
    for course_desc_1 in course_desc_1_list:
        # Get embedding for course_desc_1 and add a batch dimension
        embedding1 = course_desc_embeddings[course_desc_1].unsqueeze(0)  # Shape: [1, embedding_dim]
        # Exclude the positive match(es)
        positive_course_desc_2 = df[df['text_1'] == course_desc_1]['text_2'].tolist()
        # Build a list of candidate negative course_desc_2
        candidate_course_desc_2 = [desc for desc in course_desc_2_list if desc not in positive_course_desc_2]
        
        # Get embeddings for candidate_course_desc_2 and stack them into a tensor
        embeddings2 = torch.stack([course_desc_embeddings[desc] for desc in candidate_course_desc_2])  # Shape: [num_candidates, embedding_dim]
        
        # Compute cosine similarities
        cosine_scores = util.cos_sim(embedding1, embeddings2)  # Shape: [1, num_candidates]
        
        # Get the index of the most similar course_desc_2
        top_index = torch.argmax(cosine_scores).item()
        most_similar_desc_2 = candidate_course_desc_2[top_index]
        # Add to negative examples
        negative_examples.append([course_desc_1, most_similar_desc_2, -1])
        
    neg_df = pd.DataFrame(negative_examples, columns=['text_1', 'text_2', 'label'])
    return neg_df


# Generate negative examples
neg_df = generate_negative_examples(df)

# Combine positive and negative examples
df = pd.concat([df, neg_df]).reset_index(drop=True)
















# Simple embedding model 
# -----------------------

from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load your initial model
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
custom_model = SentenceTransformer(model_name)

# Prepare training data
train_examples = []
for _, row in df.iterrows():  # Replace `df` with your training DataFrame
    text1 = row['text_1']
    text2 = row['text_2']
    label = row['label']
    score = 1.0 if label == 1 else 0.0
    train_examples.append(InputExample(texts=[text1, text2], label=score))

# Create DataLoader
batch_size = 16
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# Define loss function
train_loss = losses.CosineSimilarityLoss(model=custom_model)

# Train the model
num_epochs = 3
warmup_steps = 100

custom_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path='initial_custom_embedding_model'
)

# Load and evaluate the initial model
initial_model = SentenceTransformer('initial_custom_embedding_model')

# Prepare validation examples
val_examples = []
for _, row in df.iterrows():  # Replace `val_df` with your validation DataFrame
    text1 = row['text_1']
    text2 = row['text_2']
    label = row['label']
    val_examples.append(InputExample(texts=[text1, text2], label=label))

# Define evaluation function
def evaluate_model(model, val_examples, threshold=0.5):
    predictions, true_labels = [], []
    
    for example in val_examples:
        # Get embeddings for both texts in the pair
        embedding1 = model.encode(example.texts[0], convert_to_tensor=True)
        embedding2 = model.encode(example.texts[1], convert_to_tensor=True)

        # Compute cosine similarity between embeddings
        cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()

        # Apply threshold to decide if the texts are similar or not
        predicted_label = 1 if cosine_score >= threshold else -1
        predictions.append(predicted_label)
        true_labels.append(int(example.label))

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='binary')
    recall = recall_score(true_labels, predictions, average='binary')
    f1 = f1_score(true_labels, predictions, average='binary')
    
    return accuracy, precision, recall, f1

# Evaluate the initial model
initial_accuracy, initial_precision, initial_recall, initial_f1 = evaluate_model(initial_model, val_examples)
print("Initial Model Performance")
print(f"Accuracy: {initial_accuracy}")
print(f"Precision: {initial_precision}")
print(f"Recall: {initial_recall}")
print(f"F1 Score: {initial_f1}")

# Fine-tune the model with new parameters
num_epochs = 4  # Increase number of epochs
learning_rate = 2e-5  # Lower learning rate
warmup_steps = 200  # Increase warmup steps

# Reinitialize DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# Use a different loss function for further fine-tuning
train_loss = losses.ContrastiveLoss(model=initial_model)

# Fine-tune the model
initial_model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    optimizer_params={'lr': learning_rate},
    output_path='fine_tuned_custom_embedding_model'
)

# Load the fine-tuned model
fine_tuned_model = SentenceTransformer('fine_tuned_custom_embedding_model')

# Evaluate the fine-tuned model
fine_tuned_accuracy, fine_tuned_precision, fine_tuned_recall, fine_tuned_f1 = evaluate_model(fine_tuned_model, val_examples)
print("\nFine-Tuned Model Performance")
print(f"Accuracy: {fine_tuned_accuracy}")
print(f"Precision: {fine_tuned_precision}")
print(f"Recall: {fine_tuned_recall}")
print(f"F1 Score: {fine_tuned_f1}")

# Compare initial and fine-tuned performance
print("\nComparison")
print(f"Accuracy Improvement: {fine_tuned_accuracy - initial_accuracy}")
print(f"Precision Improvement: {fine_tuned_precision - initial_precision}")
print(f"Recall Improvement: {fine_tuned_recall - initial_recall}")
print(f"F1 Score Improvement: {fine_tuned_f1 - initial_f1}")

























import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


train_examples = []
for _, row in df.iterrows():
    text1 = row['text_1']
    text2 = row['text_2']
    label = row['label']
    score = 1.0 if label == 1 else 0.0
    train_examples.append(InputExample(texts=[text1, text2], label=score))

# Create DataLoader
batch_size = 16
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

# Initialize the model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model=model)

# Train the model
num_epochs = 3
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=100,
    output_path='custom_embedding_model'
)

# Load and use the fine-tuned model
custom_model = SentenceTransformer('custom_embedding_model')
texts = ["Example sentence 1", "Example sentence 2"]
embeddings = custom_model.encode(texts)
print(embeddings)





from sentence_transformers import util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create validation examples
val_examples = []
for _, row in df.iterrows():  # Assuming you have a validation DataFrame `val_df`
    text1 = row['text_1']
    text2 = row['text_2']
    label = row['label']
    val_examples.append(InputExample(texts=[text1, text2], label=label))

# Encode the validation examples
predictions, true_labels = [], []
for example in val_examples:
    # Get embeddings for both texts in the pair
    embedding1 = custom_model.encode(example.texts[0], convert_to_tensor=True)
    embedding2 = custom_model.encode(example.texts[1], convert_to_tensor=True)

    # Compute cosine similarity between embeddings
    cosine_score = util.pytorch_cos_sim(embedding1, embedding2).item()

    # Apply a threshold to decide if the texts are similar or not
    predicted_label = 1 if cosine_score >= 0.5 else -1
    predictions.append(predicted_label)
    true_labels.append(example.label)

# Calculate evaluation metrics
accuracy = accuracy_score(true_labels, predictions)
# precision = precision_score(true_labels, predictions)
# recall = recall_score(true_labels, predictions)
# f1 = f1_score(true_labels, predictions)

print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")



# ---------------------------------------------
# Custom Embededings

# Path to cache file
embedding_cache_path = "embeddings/custom_embeddings_model/custom_embedding_cache.pkl"
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


batch_size = 64
modified_embedding_length = 2048
max_epochs = 10
learning_rate= 0.001
dropout_fraction =0.1
print_progress = True
save_results = True

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

        # Define cosine embedding loss function

    def cosine_embedding_loss(predictions, targets):
        # Reshape predictions and targets to 2D (batch_size, 1)
        predictions = predictions.view(-1, 1)
        targets = targets.view(-1, 1)
        similarity_targets = torch.ones(predictions.size(0))  # All set to 1 for similarity
        return torch.nn.functional.cosine_embedding_loss(predictions, targets, similarity_targets)

    # initialize projection matrix
    embedding_length = len(df["text_1_embedding"].values[0])
    matrix = torch.randn(
        embedding_length, modified_embedding_length, requires_grad=True
    )

    epochs, types, losses, accuracies, matrices = [], [], [], [], []
    for epoch in range(1, 1 + max_epochs):
        # iterate through training dataloader
        for a, b, actual_similarity in train_loader:
            # print(a, b, actual_similarity)
            # generate prediction
            predicted_similarity = model(a, b, matrix)
            # get loss and perform backpropagation
            loss = cosine_embedding_loss(predicted_similarity, actual_similarity)
            loss.backward()
            # update the weights
            with torch.no_grad():
                matrix -= matrix.grad * learning_rate
                # set gradients to zero
                matrix.grad.zero_()
        # calculate test loss
        test_predictions = model(e1_test, e2_test, matrix)
        test_loss = cosine_embedding_loss(test_predictions, s_test)

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
# imports
from typing import List, Tuple  # for type hints


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


for dataset in ["train", "test"]:
    data = df[df["dataset"] == dataset]
    a, se = accuracy_and_se(data["cosine_similarity"], data["label"])
    print(f"{dataset} accuracy: {a:0.1%} ± {1.96 * se:0.1%}")

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



results = []
max_epochs = 10
dropout_fraction = 0.1

# Try smaller batch sizes and learning rates
for batch_size, learning_rate in [(8, 0.0001), (16, 0.0001), (32, 0.001), (64, 0.0005)]:
    result = optimize_matrix(
        batch_size=batch_size,
        learning_rate=5e-5,
        max_epochs=max_epochs,
        dropout_fraction=dropout_fraction,
        save_results=False,
    )
    results.append(result)


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











# OLD CODE DO NOT USE
# ------------------------------------------------------------
# Prepare data for fine-tuning
train_examples = []
for idx, row in df.iterrows():
    example = InputExample(
        texts=[row['text_1'], row['text_2']],
        label=float(row['label'])
    )
    train_examples.append(example)

# Split into train and test sets
from sklearn.model_selection import train_test_split
train_examples, test_examples = train_test_split(train_examples, test_size=0.2, random_state=42)

# Define the DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# Define the loss function
train_loss = losses.CosineSimilarityLoss(model=model)

# Fine-tune the model
num_epochs = 1  # Adjust as needed
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)  # 10% of training steps

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps
)

# Save the fine-tuned model
model_save_path = 'fine_tuned_course_similarity_model'
model.save(model_save_path)

# Now you can use the fine-tuned model to generate embeddings

# Example: Generate embeddings for course_desc_1 and course_desc_2
df['text_1_embedding'] = df['text_1'].apply(lambda x: model.encode(x, convert_to_tensor=True))
df['text_2_embedding'] = df['text_2'].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Compute similarity scores
df['similarity_score'] = [util.cos_sim(a, b).item() for a, b in zip(df['text_1_embedding'], df['text_2_embedding'])]

# You can now use these embeddings and similarity scores for your downstream tasks

# Optionally save embeddings to a file
# df.to_csv('data_with_embeddings.csv', index=False)
