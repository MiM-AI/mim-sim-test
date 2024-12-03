import pandas as pd
import numpy as np
import gzip
import pickle

from app import find_most_similar_courses


institutions_dict = {"Portland Community College (PCC)": "PCC",
				"Portland State University": "PSU", 
				"Amherst College": "AC"}

institutions_dict = {"Portland Community College (PCC)": "PCC",
				"Portland State University": "PSU", 
				"Amherst College": "AC"}

institutions = {"Central-Oregon-Community-College",
"Chemeketa-Community-College",
"Clackamas-Community-College",
"Lane-Community-College",
"Linn-Benton-Community-College",
"Mt-Hood-Community-College",
"Oregon-Institute-of-Technology",
"Portland-Community-College",
"Portland-State-University",
"Western-Oregon-University"}


def get_upper_lower_dat(institutions):

	# institutions = [x for x in institutions_dict]

	# Load upper/lower division transfer
	udat = pd.read_excel("data/Lower_to_Upper_Transfer.xlsx")

	udat['Institution'] = udat['Institution'].str.replace(" ", "-")
	udat['Institution'] = udat['Institution'].str.replace(" (PCC)", "")
	
	udat = udat[udat['Institution'].isin(institutions)]

	udat[udat['Institution'].str.contains("Oregon")]

	# [x for x in institutions if x not in udat['Institution'].unique()]

	# udat = udat[udat['Transfer Term'] >= 202000]

	# udat['Institution'] = udat['Institution'].replace(institutions_dict)

	start = np.floor(udat['Transfer Term']/100).astype(int) - 1 
	end = np.floor(udat['Transfer Term']/100).astype(int) 
	udat['year'] = start.astype(str) + "-" + end.astype(str)

	udat['course_code'] = udat['Transfer Subject'] + " " + udat['Transfer Course'].astype(str)

	udat = udat[['Institution', 'year', 'course_code', 'OSU Course']].reset_index(drop=True)

	udat.columns = ['Ext_Inst', 'Transfer_Year', 'Ext_Course_Code', 'Int_Course_Code']

	return udat


def get_articulation_dat(external_institution):
	print(f"Processing: {external_institution}")
	external_institution = external_institution.replace("-", " ")
	
	adat = pd.read_csv('data/Articulation_data.csv', encoding="ISO-8859-1", low_memory=False)
	adat['TRNS_DESCRIPTION'] = adat['TRNS_DESCRIPTION'].str.replace(" (PCC)", "")


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


def get_art_tables(institutions):
	institutions = [x.replace("-", " ") for x in institutions]

	adat = [get_articulation_dat(x) for x in institutions]
	mdat = pd.concat(adat)

	[x for x in institutions if x not in mdat['TRNS_DESCRIPTION'].unique()]

	# adat[adat['TRNS_DESCRIPTION'].str.contains("Oregon")]

	# len(mdat['TRNS_DESCRIPTION'].unique())

	# mdat['TRNS_DESCRIPTION'] = mdat['TRNS_DESCRIPTION'].replace(institutions_dict)

	# mdat = mdat[mdat['YEAR'].astype(int) >= 2020]

	start = mdat['YEAR'].astype(int) - 1 
	end =  mdat['YEAR'].astype(int) 
	mdat['year'] = start.astype(str) + "-" + end.astype(str)

	mdat = mdat[['TRNS_DESCRIPTION', 'year', 'EXT COURSE CODE', 'INT COURSE CODE']]

	mdat.columns = ['Ext_Inst', 'Transfer_Year', 'Ext_Course_Code', 'Int_Course_Code']

	return mdat.reset_index(drop=True)


def proc_acc(udat, apply_matrix=False, keywords=False, multimodel=False):
	# cims = pd.read_excel("data/CIM approved snapshot 2024-07-15.xlsx")

    with gzip.open(f"embeddings/OSU/osu_course_embeddings.pkl.gz", "rb") as f:
    	target_df = pickle.load(f)

    target_df.columns = ['COURSE CODE', 'COURSE TITLE', 'DESCRIPTION', 'CODE TITLE DESC', 'embedding']

	with gzip.open(f"embeddings/Oregon-State-University/2023-2024.pkl.gz", "rb") as f:
    	cims = pickle.load(f)

    cims = pd.concat([target_df, cims])
    cims = cims.drop_duplicates(subset=['COURSE CODE'])

	# Iterate through udat and update with match information
	matches = []

	row = udat.iloc[0, :]
	row
	for idx, row in udat.iterrows():
		# print(row)
        if multimodel == True:
	        external_course_info, similar_courses1 = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], apply_matrix=apply_matrix, keywords=True)
	        external_course_info, similar_courses2 = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], apply_matrix=apply_matrix, keywords=False)

	        if isinstance(similar_courses1, pd.DataFrame) and isinstance(similar_courses2, pd.DataFrame):

		        # Merge the two dataframes on 'code'
		        merged_df = pd.merge(similar_courses1, similar_courses2, on='COURSE CODE', suffixes=('_1', '_2'))

		        # Compute weighted average of similarity scores
		        merged_df['weighted_similarity'] = (merged_df['similarity_score_1'] + merged_df['similarity_score_2']) / 2

		        # Sort the dataframe by weighted similarity score in descending order
		        similar_courses = merged_df[['COURSE CODE', 'COURSE TITLE_1', 'DESCRIPTION_1', 'CODE TITLE DESC_1', 'weighted_similarity']].sort_values(by='weighted_similarity', ascending=False)
		        similar_courses.columns = ['COURSE CODE', 'COURSE TITLE', 'DESCRIPTION', 'CODE TITLE DESC', 'similarity_score']

		else:
			# Find most similar courses
			_, similar_courses = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], 
									top_n=20, apply_matrix=apply_matrix, keywords=keywords)
		
		if isinstance(similar_courses, pd.DataFrame):

			similar_courses['sim_row'] = range(1, len(similar_courses) + 1)
			similar_courses['COURSE CODE'] = similar_courses['COURSE CODE'].str.replace(r'(\d+)[^\d]*$', r'\1', regex=True)

			# Check for match
			if row['Int_Course_Code'] in similar_courses['COURSE CODE'].values:
				# Find the row position and similarity score in similar_courses
				match_row = similar_courses[similar_courses['COURSE CODE'] == row['Int_Course_Code']].iloc[0]
				
				matches.append({
					'Ext_Course_Code': row['Ext_Course_Code'],
					'OSU_Course_Code': match_row['COURSE CODE'],
					'row_number': match_row.sim_row,  # Original row number from udat
					'match': 1,
					'similar_courses_row_number': match_row.name,  # Row position from similar_courses
					'similarity_score': match_row['similarity_score']
				})
				print("Match found!")
			else:   # If it doesn't match, check to see if couse is available
				check = cims[cims['COURSE CODE'] == row['Int_Course_Code']]
				match = np.where(len(check) > 0, 0, -1)
				matches.append({
					'Ext_Course_Code': row['Ext_Course_Code'],
					'OSU_Course_Code': None,
					'row_number': None,
					'match': match,
					'similar_courses_row_number': None,
					'similarity_score': None
				})

		else:
			matches.append({
				'Ext_Course_Code': row['Ext_Course_Code'],
				'OSU_Course_Code': None,
				'row_number': None,
				'match': -1,
				'similar_courses_row_number': None,
				'similarity_score': None
			})

	# Convert matches to DataFrame and merge back to udat
	matches_df = pd.DataFrame(matches)
	outdat = pd.concat([udat, matches_df], axis=1)
	
	print_df = outdat[outdat['match'] != -1]
	len_outdat = len(print_df)

	print(f"n = {len_outdat}")
	for i in range(1, int(print_df['row_number'].max()) + 1):
		# print(i)
		row_len = len(print_df[print_df['row_number'] <= i])
		print(f"Matches within top {i}: {np.round((row_len/len_outdat)*100, 2)}%")

	return outdat


udat1 = get_upper_lower_dat(institutions)
udat2 = get_art_tables(institutions)
udat = pd.concat([udat1, udat2]).reset_index(drop=True)

udat['Transfer_Year'].split("-")

udat["Year"]

# udat['Ext_Inst'] = udat['Ext_Inst'].str.replace(" ", "-")
# test = udat[udat['Ext_Inst'].str.contains("Lane-Comm")]
# test[test['Ext_Course_Code'].str.contains("BI")]

# test = pdat[pdat['TRNS_DESCRIPTION'].str.contains("Lane Comm")]
# test[test['EXT COURSE CODE'].str.contains("BI 231")]


outdat = proc_acc(udat, False, False, False)

outdat2 = outdat.dropna(subset=['similarity_score'])

outdat2['similarity_score']

outdat = proc_acc(udat, False, False, True)



# Matches within top 1: 42.22%
# Matches within top 2: 61.11%
# Matches within top 3: 68.89%
# Matches within top 4: 78.89%
# Matches within top 5: 82.22%
# Matches within top 6: 85.56%
# Matches within top 7: 87.78%
# Matches within top 8: 88.89%
# Matches within top 9: 90.0%
# Matches within top 10: 91.11%
# Matches within top 11: 94.44%
# Matches within top 12: 95.56%
# Matches within top 13: 95.56%
# Matches within top 14: 95.56%
# Matches within top 15: 95.56%
# Matches within top 16: 95.56%
# Matches within top 17: 96.67%


proc_acc(udat, True, True)

apply_matrix=False
keywords=False


# Check multimodel
output = proc_acc(udat, False, False, True)



with gzip.open(f"embeddings/OSU/osu_course_embeddings.pkl.gz", "rb") as f:
	target_df = pickle.load(f)

cims = pd.read_excel("data/CIM approved snapshot 2024-07-15.xlsx")

target_df[target_df['code'] == 'CS 361']
target_df[target_df['code'].str.contains('CS 3')]

cims[cims['Course Code (code)'] == 'PH 206']
cims[cims['Course Code (code)'].str.contains('PH 20')]



output = proc_acc(udat)


def proc_acc(udat, apply_matrix=False, keywords=False, model_weights=None):
	"""
	Process accuracy by comparing external courses to internal courses.

	Parameters:
	- udat (DataFrame): Input DataFrame.
	- apply_matrix (bool): Whether to apply a similarity matrix.
	- keywords (bool): Whether to use keyword-based matching.
	- model_weights (dict): Weights for each model, e.g., {'model1': 0.6, 'model2': 0.4}.

	Returns:
	- outdat (DataFrame): Updated DataFrame with match information.
	"""
	cims = pd.read_excel("data/CIM approved snapshot 2024-07-15.xlsx")
	matches = []

	# Set default weights if none are provided
	if model_weights is None:
		model_weights = {'model1': 0.5, 'model2': 0.5}

	for idx, row in udat.iterrows():
		print(idx)
		# Find most similar courses (simulating results from both models)
		_, similar_courses_model1 = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], 
															  apply_matrix=apply_matrix, keywords=False)
		_, similar_courses_model2 = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], 
															  apply_matrix=apply_matrix, keywords=True)
		
		# Merge or align results from both models
		if isinstance(similar_courses_model1, pd.DataFrame) and isinstance(similar_courses_model2, pd.DataFrame):
			similar_courses_model1['sim_row'] = range(1, len(similar_courses_model1) + 1)
			similar_courses_model2['sim_row'] = range(1, len(similar_courses_model2) + 1)

			# Align by course code for weighted averaging
			similar_courses = pd.merge(
				similar_courses_model1.rename(columns={'similarity_score': 'similarity_score_model1'}),
				similar_courses_model2.rename(columns={'similarity_score': 'similarity_score_model2'}),
				on='code',
				how='outer'
			)

			# Compute weighted similarity
			similar_courses['weighted_similarity'] = (
				model_weights['model1'] * similar_courses['similarity_score_model1'].fillna(0) +
				model_weights['model2'] * similar_courses['similarity_score_model2'].fillna(0)
			)

			# Sort by weighted similarity
			similar_courses = similar_courses.sort_values('weighted_similarity', ascending=False)
			similar_courses.reset_index(drop=True, inplace=True)

			if row['Int_Course_Code'] in similar_courses['code'].values:
				match_row = similar_courses[similar_courses['code'] == row['Int_Course_Code']].iloc[0]
				matches.append({
					'Ext_Course_Code': row['Ext_Course_Code'],
					'OSU_Course_Code': match_row['code'],
					'row_number': match_row['sim_row_x'] if 'sim_row_x' in match_row else None,
					'match': 1,
					'similar_courses_row_number': match_row.name,
					'similarity_score': match_row['weighted_similarity']
				})
			else:
				# No match found; check in approved courses
				check = cims[cims['Course Code (code)'] == row['Int_Course_Code']]
				match = 0 if len(check) > 0 else -1
				matches.append({
					'Ext_Course_Code': row['Ext_Course_Code'],
					'OSU_Course_Code': None,
					'row_number': None,
					'match': match,
					'similar_courses_row_number': None,
					'similarity_score': None
				})
		else:
			matches.append({
				'Ext_Course_Code': row['Ext_Course_Code'],
				'OSU_Course_Code': None,
				'row_number': None,
				'match': -1,
				'similar_courses_row_number': None,
				'similarity_score': None
			})

	# Convert matches to DataFrame and merge back to udat
	matches_df = pd.DataFrame(matches)
	outdat = pd.concat([udat, matches_df], axis=1)

	# Print summary statistics
	print_df = outdat[outdat['match'] != -1]
	len_outdat = len(print_df)

	print(f"n = {len_outdat}")
	for i in range(1, int(print_df['row_number'].max()) + 1):
		row_len = len(print_df[print_df['row_number'] <= i])
		print(f"Matches within top {i}: {np.round((row_len / len_outdat) * 100, 2)}%")

	return outdat





