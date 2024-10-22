import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import gzip


# external_institution = "Portland State University"
# years = "2023-2024"
# course_code = "Soc 456"
# target_df = internal_emb

# test = find_most_similar_courses(institution, years, course_code, target_df, top_n=10)

def find_most_similar_courses(institution, years, course_code, target_df, top_n=10):
    """
    Finds the top N most similar courses based on cosine similarity.

    Parameters:
    - selected_embedding: array-like, the embedding of the course to compare
    - target_df: DataFrame, contains course information and their embeddings
    - top_n: int, the number of top similar courses to return

    Returns:
    - DataFrame of top N similar courses with their similarity scores
    """
    # Load the pickle file
    if external_institution == "Portland Community College":
        institution = "PCC"
    elif external_institution == "Portland State University":
        institution = "PSU"
    elif external_institution == "Amherst College":
        institution = "AC"
    try:
        # with open(f"embeddings/{institution}/{years}.pkl", 'rb') as f:
        #     selected_embedding = pickle.load(f)
        with gzip.open(f"embeddings/{institution}/{years}.pkl.gz", "rb") as f:
            selected_embedding = pickle.load(f)

    except FileNotFoundError:
        st.error("The embedding file for this institution and year could not be found.")
        return None

    # Check if the course code ends with a letter
    if re.search(r'\d+[A-Z]$', course_code):
        # Remove the letter at the end of the course code
        course_code = course_code[:-1]

    course_code = course_code.upper()

    selected_embedding['COURSE CODE'] = selected_embedding['COURSE CODE'].str.upper()
    selected_embedding = selected_embedding.dropna(subset=['COURSE CODE']).reset_index(drop=True)
    selected_embedding = selected_embedding[selected_embedding['COURSE CODE'] != 'N/A']

    selected_embedding = selected_embedding[selected_embedding['COURSE CODE'].str.contains(course_code)].reset_index(drop=True)
    # Extract specific course info for external_course_info
    external_course_info = selected_embedding[['COURSE CODE', 'COURSE TITLE', 'DESCRIPTION']]
    
    print(external_course_info)
    # test['COURSE CODE'].unique()

    # selected_embedding[selected_embedding['COURSE CODE'].str.contains("ENG")]
    # selected_embedding[selected_embedding['COURSE CODE'].str.contains("ENG")]['COURSE CODE'].unique()
    # selected_embedding[selected_embedding['COURSE CODE'].str.contains(course_code)].reset_index(drop=True)

    # 4453 ANTH 340U Design, Politics and Society  Anthropological approaches to design aesthetic...

    # selected_embedding = selected_embedding[selected_embedding['COURSE CODE'] == course_code].reset_index(drop=True)
    # selected_embedding = selected_embedding[selected_embedding['COURSE TITLE'] == course_title]
    
    if len(selected_embedding) == 0:
        print(f"{course_code} not found")
        return None

    selected_embedding = selected_embedding.loc[0, 'embedding']

    # Reshape the selected course embedding for computation
    selected_embedding = np.array(selected_embedding).reshape(1, -1)
    
    # Stack the embeddings from the target DataFrame
    target_embeddings = np.vstack(target_df['embedding'])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(selected_embedding, target_embeddings).flatten()
    
    # Get the indices of the top N most similar courses
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Retrieve the top N similar courses and add the similarity score
    similar_courses = target_df.iloc[top_indices].copy()
    similar_courses['similarity_score'] = similarities[top_indices]
    
    return external_course_info, similar_courses



def find_most_similar_courses(institution, years, course_code, target_df, top_n=10):
    """
    Finds the top N most similar courses based on cosine similarity.
    """

    if external_institution == "Portland Community College":
        institution = "PCC"
    elif external_institution == "Portland State University":
        institution = "PSU"
    elif external_institution == "Amherst College":
        institution = "AC"

    try:
        with gzip.open(f"embeddings/{institution}/{years}.pkl.gz", "rb") as f:
            selected_embedding = pickle.load(f)

    except FileNotFoundError:
        st.error("The embedding file for this institution and year could not be found.")
        return None

    # Process course code
    if re.search(r'\d+[A-Z]$', course_code):
        course_code = course_code[:-1]
    course_code = course_code.upper()

    # Filter and clean selected embedding
    selected_embedding['COURSE CODE'] = selected_embedding['COURSE CODE'].str.upper()
    selected_embedding = selected_embedding.dropna(subset=['COURSE CODE']).reset_index(drop=True)
    selected_embedding = selected_embedding[selected_embedding['COURSE CODE'] != 'N/A']
    selected_embedding = selected_embedding[selected_embedding['COURSE CODE'].str.contains(course_code)].reset_index(drop=True)
    
    # Return None if no matching course is found
    if len(selected_embedding) == 0:
        print(f"{course_code} not found")
        return None, None
    
    # Extract specific course info for external_course_info
    external_course_info = selected_embedding[['COURSE CODE', 'COURSE TITLE', 'DESCRIPTION']]
    
    # Extract embedding for similarity check
    selected_embedding = selected_embedding.loc[0, 'embedding']
    selected_embedding = np.array(selected_embedding).reshape(1, -1)

    # Prepare target embeddings and calculate similarities
    target_embeddings = np.vstack(target_df['embedding'])
    similarities = cosine_similarity(selected_embedding, target_embeddings).flatten()
    
    # Select the top N indices based on similarity
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    similar_courses = target_df.iloc[top_indices].copy()
    similar_courses['similarity_score'] = similarities[top_indices]

    return external_course_info, similar_courses




def load_courses(institution, years):
    try:
        # Load the embedding data for the selected institution and year
        # with open(f"embeddings/{institution}/{years}.pkl", 'rb') as f:
        #     data = pickle.load(f)

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
    available_courses = load_courses(institution, years)
    if available_courses:
        course_code = st.selectbox("Select Course Code", list(available_courses.keys()), format_func=lambda x: f"{x} - {available_courses[x]}")
    else:
        st.warning("No courses available for the selected institution and year.")
else:
    course_code = None

# Load the target DataFrame (e.g., from a file)
# Load the pickle file
# with open(f"embeddings/OSU/osu_course_embeddings.pkl", 'rb') as f:
#     internal_emb = pickle.load(f)

with gzip.open(f"embeddings/OSU/osu_course_embeddings.pkl.gz", "rb") as f:
    internal_emb = pickle.load(f)


# Button to perform similarity check
if st.button("Find Similar Courses"):
    if course_code:
        # print(course_code)
        # print(external_institution)

        external_course_info, similar_courses = find_most_similar_courses(external_institution, years, course_code, internal_emb)
        print("EXTERNAL COURSE INFO")
        print(external_course_info)
        print("SIMILAR COURSES")
        print(similar_courses)
        if similar_courses is not None:
            st.write("### External Course Info:")
            st.write(f"{external_course_info['COURSE CODE'].iat[0]} - {external_course_info['COURSE TITLE'].iat[0]}")
            st.write(f"{external_course_info['DESCRIPTION'].iat[0]}")
            st.write("---")  # Separator for readability
            st.write("\n")  # Separator for readability

            st.write("### Top OSU Similar Courses:")
            # Loop through each row in the DataFrame and format the output
            for idx, row in similar_courses.iterrows():
                # print(row)
                # print(row['code'])
                similarity_score = f"{row['similarity_score'] * 100:.2f}%"
                print(similarity_score)
                
                # st.write(f"**Similar {idx + 1} - Similarity Index: {similarity_score}**")
                st.write(f"{row['code']} - {row['title']} (Similarity Index: {similarity_score})")
                st.write(f"**Description**: {row['description']}")
                st.write("---")  # Separator for readability
            
        else:
            st.write("No similar courses found.")
    else:
        st.warning("Please select a course code to search.")




