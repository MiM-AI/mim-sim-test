import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import gzip
import torch  # for matrix optimization
from typing import List, Tuple  # for type hints

external_institution = "Portland Community College"
years = "2019-2020"
course_code = "Soc 204"
# target_df = internal_emb
# matrix = best_matrix_loaded

# test = find_most_similar_courses(institution, years, course_code, best_matrix_loaded, target_df, top_n=10)

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

def embedding_multiplied_by_matrix(
    embedding: List[float], matrix: torch.tensor
) -> np.array:
    embedding_tensor = torch.tensor(embedding).float()
    modified_embedding = embedding_tensor @ matrix
    modified_embedding = modified_embedding.detach().numpy()
    return modified_embedding


def find_most_similar_courses(institution, years, course_code, top_n=10, apply_matrix=True, keywords=True):
    """
    Finds the top N most similar courses based on cosine similarity, using custom embeddings if available.
    
    Parameters:
    - institution (str): Name of the institution.
    - years (str): Year or range of years for the embedding file.
    - course_code (str): The course code to match.
    - target_df (DataFrame): DataFrame containing target course embeddings.
    - matrix (ndarray): Transformation matrix to adjust embeddings.
    - top_n (int): Number of top similar courses to return.
    - apply_matrix (bool): Whether to apply the matrix to embeddings.
    
    Returns:
    - external_course_info (DataFrame): DataFrame with course code, title, and description.
    - similar_courses (DataFrame): DataFrame with top N similar courses and their similarity scores.
    """

    # Handle specific institution naming
    if institution == "Portland Community College":
        institution = "PCC"
    elif institution == "Portland State University":
        institution = "PSU"
    elif institution == "Amherst College":
        institution = "AC"

    try:
        if keywords == True:
            with gzip.open(f"embeddings/{institution}/{years}_keywords.pkl.gz", "rb") as f:
                selected_embedding = pickle.load(f)

            with open("embeddings/custom_embeddings_model/best_matrix_keywords.pkl", "rb") as f:
                matrix = pickle.load(f)

            with gzip.open(f"embeddings/OSU/osu_course_embeddings_keywords.pkl.gz", "rb") as f:
                target_df = pickle.load(f)

        else:
            with gzip.open(f"embeddings/{institution}/{years}.pkl.gz", "rb") as f:
                selected_embedding = pickle.load(f)

            with open("embeddings/custom_embeddings_model/best_matrix.pkl", "rb") as f:
                matrix = pickle.load(f)

            with gzip.open(f"embeddings/OSU/osu_course_embeddings.pkl.gz", "rb") as f:
                target_df = pickle.load(f)

    except FileNotFoundError:
        print("The embedding file for this institution and year could not be found.")
        return None, None

    # Process course code
    if re.search(r'\d+[A-Z]$', course_code):
        course_code = course_code[:-1]
    course_code = course_code.upper()

    # Filter and clean selected embedding
    selected_embedding['COURSE CODE'] = selected_embedding['COURSE CODE'].str.upper()
    selected_embedding = selected_embedding.dropna(subset=['COURSE CODE']).reset_index(drop=True)
    selected_embedding = selected_embedding[selected_embedding['COURSE CODE'] != 'N/A']
    # selected_embedding = selected_embedding[selected_embedding['COURSE CODE'].str.contains(course_code)].reset_index(drop=True)
    selected_embedding = selected_embedding[selected_embedding['COURSE CODE'] == course_code].reset_index(drop=True)

    # Return None if no matching course is found
    if len(selected_embedding) == 0:
        print(f"{course_code} not found")
        return None, None
    
    # Extract specific course info for external_course_info
    external_course_info = selected_embedding[['COURSE CODE', 'COURSE TITLE', 'DESCRIPTION']]
    
    # Extract and optionally adjust embedding with custom matrix for similarity check
    selected_embedding = selected_embedding.loc[0, 'embedding']
    
    if apply_matrix:
        selected_embedding = embedding_multiplied_by_matrix(np.array(selected_embedding), matrix).reshape(1, -1)
    else:
        selected_embedding = np.array(selected_embedding).reshape(1, -1)

    # Prepare target embeddings, optionally apply matrix, and calculate similarities
    if apply_matrix:
        target_embeddings = np.vstack(target_df['embedding'].apply(lambda x: embedding_multiplied_by_matrix(x, matrix)))
    else:
        target_embeddings = np.vstack(target_df['embedding'].apply(np.array))
    
    similarities = cosine_similarity(selected_embedding, target_embeddings).flatten()
    
    # Select the top N indices based on similarity
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    similar_courses = target_df.iloc[top_indices].copy()
    similar_courses['similarity_score'] = similarities[top_indices]

    return external_course_info, similar_courses


def load_courses(institution, years, keywords):
    try:
        # Load the embedding data for the selected institution and year
        # with open(f"embeddings/{institution}/{years}.pkl", 'rb') as f:
        #     data = pickle.load(f)

        if keywords == True:
            with gzip.open(f"embeddings/{institution}/{years}_keywords.pkl.gz", "rb") as f:
                data = pickle.load(f)
        else:
            with gzip.open(f"embeddings/{institution}/{years}.pkl.gz", "rb") as f:
                data = pickle.load(f)

        # Extract course codes and titles
        courses = data[['COURSE CODE', 'COURSE TITLE']].drop_duplicates()
        courses = courses.sort_values('COURSE CODE')
        course_dict = dict(zip(courses['COURSE CODE'], courses['COURSE TITLE']))
        return course_dict
    except FileNotFoundError:
        st.warning(f"Data for {institution} in {years} not found.")
        return {}

        return course_dict
    
    except FileNotFoundError:
        st.warning(f"Data for {institution} in {years} not found.")
        return {}


if __name__ == "__main__":

    st.title("Course Similarity Checker")

    # User inputs
    external_institution = st.selectbox("Select Institution", ["Portland Community College", "Portland State University", "Amherst College"])

    years = st.selectbox("Select Year", ["2019-2020", "2020-2021", "2021-2022", "2022-2023", "2023-2024"])

    # Load the pickle file
    if external_institution == "Portland Community College":
        institution = "PCC"
    elif external_institution == "Portland State University":
        institution = "PSU"
    elif external_institution == "Amherst College":
        institution = "AC"


    # Load courses when both institution and year are selected
    if institution and years:
        available_courses = load_courses(institution, years, keywords=True)
        if available_courses:
            course_code = st.selectbox("Select Course Code", list(available_courses.keys()), format_func=lambda x: f"{x} - {available_courses[x]}")
        else:
            st.warning("No courses available for the selected institution and year.")
    else:
        course_code = None

    # apply_matrix = st.checkbox("Apply Custom Embeddings", value=False)
    # keywords = st.checkbox("Keyword Embeddings", value=False)
    # multimodel = st.checkbox("Multi-model (Baseline + Keyword)", value=False)
    multimodel = False
    keywords = False
    apply_matrix = False

    # Button to perform similarity check
    if st.button("Find Similar Courses"):
        if course_code:
            # print(course_code)
            # print(external_institution)
            if multimodel == True:
                external_course_info, similar_courses1 = find_most_similar_courses(external_institution, years, course_code, apply_matrix=apply_matrix, keywords=True)
                external_course_info, similar_courses2 = find_most_similar_courses(external_institution, years, course_code, apply_matrix=apply_matrix, keywords=False)

                # Merge the two dataframes on 'code'
                merged_df = pd.merge(similar_courses1, similar_courses2, on='code', suffixes=('_1', '_2'))

                # Compute weighted average of similarity scores
                merged_df['weighted_similarity'] = (merged_df['similarity_score_1'] + merged_df['similarity_score_2']) / 2

                # Sort the dataframe by weighted similarity score in descending order
                similar_courses = merged_df[['code', 'title_1', 'description_1', 'CODE TITLE DESC_1', 'weighted_similarity']].sort_values(by='weighted_similarity', ascending=False)
                similar_courses.columns = ['code', 'title', 'description', 'CODE TITLE DESC', 'similarity']

            else:
                external_course_info, similar_courses = find_most_similar_courses(external_institution, years, course_code, apply_matrix=apply_matrix, keywords=keywords)
            
            print("--------------------------------")
            print("EXTERNAL COURSE INFO:")
            # print(external_course_info)
            print(external_course_info['COURSE CODE'].iat[0], external_course_info['COURSE TITLE'].iat[0])
            print("--------------------------------")
            print("SIMILAR COURSES")
            print(similar_courses)
            if similar_courses is not None:
                st.write("### External Course Info:")
                st.write(f"{external_course_info['COURSE CODE'].iat[0]} - {external_course_info['COURSE TITLE'].iat[0]}")
                st.write(f"{external_course_info['DESCRIPTION'].iat[0]}")
                st.write("---")  # Separator for readability

                st.write("### Top OSU Similar Courses:")
                # Loop through each row in the DataFrame and format the output
                for idx, row in similar_courses.iterrows():
                    similarity_score = row.get('similarity_score', row.get('similarity_score_custom', None))
                    if similarity_score is not None:
                        similarity_score = f"{similarity_score * 100:.2f}%"
                        # st.write(f"{row['code']} - {row['title']} (Similarity Index: {similarity_score})")
                        st.write(f"{row['code']} - {row['title']}")
                        st.write(f"**Description**: {row['description']}")
                        st.write("---")  # Separator for readability
                    else:
                        st.write(f"{row['code']} - {row['title']}")
                        st.write(f"**Description**: {row['description']}")
                        st.write("---")  # Separator for readability
            else:
                st.write("No similar courses found.")




