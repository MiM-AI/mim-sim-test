import pandas as pd
import numpy as np
import gzip
import pickle

from app import find_most_similar_courses


institutions_dict = {"Portland Community College (PCC)": "PCC",
				"Portland State University": "PSU", 
				"Amherst College": "AC"}

def get_upper_lower_dat(institutions_dict):

	institutions = [x for x in institutions_dict]

	# Load upper/lower division transfer
	udat = pd.read_excel("data/Lower_to_Upper_Transfer.xlsx")

	udat = udat[udat['Institution'].isin(institutions)]
	udat = udat[udat['Transfer Term'] >= 202000]

	udat['Institution'] = udat['Institution'].replace(institutions_dict)

	start = np.floor(udat['Transfer Term']/100).astype(int) - 1 
	end = np.floor(udat['Transfer Term']/100).astype(int) 
	udat['year'] = start.astype(str) + "-" + end.astype(str)

	udat['course_code'] = udat['Transfer Subject'] + " " + udat['Transfer Course'].astype(str)

	udat = udat[['Institution', 'year', 'course_code', 'OSU Course']].reset_index(drop=True)

	udat.columns = ['Ext_Inst', 'Transfer_Year', 'Ext_Course_Code', 'Int_Course_Code']

	return udat


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


def get_art_tables(institutions_dict):
	institutions = [x for x in institutions_dict]

	adat = [get_articulation_dat(x) for x in institutions_dict]
	mdat = pd.concat(adat)

	mdat['TRNS_DESCRIPTION'] = mdat['TRNS_DESCRIPTION'].replace(institutions_dict)

	mdat = mdat[mdat['YEAR'].astype(int) >= 2020]

	start = mdat['YEAR'].astype(int) - 1 
	end =  mdat['YEAR'].astype(int) 
	mdat['year'] = start.astype(str) + "-" + end.astype(str)

	mdat = mdat[['TRNS_DESCRIPTION', 'year', 'EXT COURSE CODE', 'INT COURSE CODE']]

	mdat.columns = ['Ext_Inst', 'Transfer_Year', 'Ext_Course_Code', 'Int_Course_Code']

	return mdat.reset_index(drop=True)


def proc_acc(udat, apply_matrix=False, keywords=False, multimodel=False):
	cims = pd.read_excel("data/CIM approved snapshot 2024-07-15.xlsx")
	
	# Iterate through udat and update with match information
	matches = []

	# row = udat.iloc[9, :]
	# row
	for idx, row in udat.iterrows():
		print(idx)
        if multimodel == True:
	        external_course_info, similar_courses1 = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], apply_matrix=apply_matrix, keywords=True)
	        external_course_info, similar_courses2 = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], apply_matrix=apply_matrix, keywords=False)

	        if isinstance(similar_courses1, pd.DataFrame) and isinstance(similar_courses2, pd.DataFrame):

		        # Merge the two dataframes on 'code'
		        merged_df = pd.merge(similar_courses1, similar_courses2, on='code', suffixes=('_1', '_2'))

		        # Compute weighted average of similarity scores
		        merged_df['weighted_similarity'] = (merged_df['similarity_score_1'] + merged_df['similarity_score_2']) / 2

		        # Sort the dataframe by weighted similarity score in descending order
		        similar_courses = merged_df[['code', 'title_1', 'description_1', 'CODE TITLE DESC_1', 'weighted_similarity']].sort_values(by='weighted_similarity', ascending=False)
		        similar_courses.columns = ['code', 'title', 'description', 'CODE TITLE DESC', 'similarity_score']

		else:
			# Find most similar courses
			_, similar_courses = find_most_similar_courses(row['Ext_Inst'], row['Transfer_Year'], row['Ext_Course_Code'], 
									apply_matrix=apply_matrix, keywords=keywords)
		
		if isinstance(similar_courses, pd.DataFrame):

			similar_courses['sim_row'] = range(1, len(similar_courses) + 1)
			similar_courses['code'] = similar_courses['code'].str.replace(r'(\d+)[^\d]*$', r'\1', regex=True)

			# Check for match
			if row['Int_Course_Code'] in similar_courses['code'].values:
				# Find the row position and similarity score in similar_courses
				match_row = similar_courses[similar_courses['code'] == row['Int_Course_Code']].iloc[0]
				
				matches.append({
					'Ext_Course_Code': row['Ext_Course_Code'],
					'OSU_Course_Code': match_row['code'],
					'row_number': match_row.sim_row,  # Original row number from udat
					'match': 1,
					'similar_courses_row_number': match_row.name,  # Row position from similar_courses
					'similarity_score': match_row['similarity_score']
				})
			else:   # If it doesn't match, check to see if couse is available
				check = cims[cims['Course Code (code)'] == row['Int_Course_Code']]
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



#    Ext_Inst Transfer_Year Ext_Course_Code Int_Course_Code  ... row_number match  similar_courses_row_number  similarity_score
# 0       PCC     2021-2022         ART 115         ART 115  ...        1.0     1                       357.0          0.825784
# 3       PCC     2019-2020         ATH 101        ANTH 240  ...        1.0     1                       282.0          0.724298
# 4       PCC     2021-2022          BA 211          BA 211  ...        1.0     1                       566.0          0.778920
# 5       PCC     2019-2020          ES 101          ES 101  ...        1.0     1                      1835.0          0.910654
# 6       PCC     2019-2020           G 201         GEO 201  ...        1.0     1                      2264.0          0.732905
# 7       PCC     2019-2020           G 202         GEO 202  ...        1.0     1                      2266.0          0.718301
# 8       PCC     2019-2020           G 203         GEO 203  ...        1.0     1                      2267.0          0.739299
# 9       PCC     2020-2021         GEO 266        GEOG 362  ...        NaN     0                         NaN               NaN
# 10      PCC     2022-2023         GEO 266        GEOG 460  ...        5.0     1                      2370.0          0.693428
# 11      PCC     2021-2022         HST 102         HST 102  ...        1.0     1                      2664.0          0.819601
# 12      PCC     2022-2023         HST 121         HST 104  ...        1.0     1                      2666.0          0.812507
# 13      PCC     2021-2022         MTH 111         MTH 111  ...        1.0     1                      3056.0          0.816988
# 14      PCC     2021-2022         MTH 112         MTH 112  ...        1.0     1                      3057.0          0.838751
# 16      PCC     2020-2021         MUS 105         MUS 101  ...        1.0     1                      3216.0          0.791413
# 17      PCC     2020-2021         PHY 101          PH 106  ...        NaN     0                         NaN               NaN
# 18      PCC     2020-2021         PHY 122          PH 206  ...        NaN     0                         NaN               NaN
# 19      PCC     2020-2021         PHY 123          PH 207  ...        NaN     0                         NaN               NaN
# 22      PCC     2021-2022         SOC 204         SOC 204  ...        1.0     1                      4128.0          0.878372
# 23      PCC     2020-2021          WR 121          WR 121  ...        1.0     1                      4504.0          0.778778
# 24      PSU     2019-2020          BA 325          BA 370  ...        1.0     1                       601.0          0.666456
# 25      PSU     2020-2021          BI 427          BI 454  ...        NaN     0                         NaN               NaN
# 26      PSU     2022-2023          CS 314          CS 361  ...        NaN     0                         NaN               NaN
# 27      PSU     2019-2020         DES 121          GD 226  ...        1.0     1                      2234.0          0.652121
# 28      PSU     2019-2020         DES 210         ART 121  ...        4.0     1                       359.0          0.625320
# 29      PSU     2019-2020         DES 290          GD 269  ...        1.0     1                      2237.0          0.699179
# 30      PSU     2020-2021        GEOG 230        GEOG 103  ...        2.0     1                      2332.0          0.769460
# 31      PSU     2019-2020         MTH 255         MTH 255  ...        2.0     1                      3070.0          0.776020
# 32      PSU     2020-2021         PSY 204         PSY 202  ...        1.0     1                      3875.0          0.756955


#    Ext_Inst Transfer_Year Ext_Course_Code Int_Course_Code Ext_Course_Code OSU_Course_Code  row_number  match  similar_courses_row_number  similarity_score
# 0       PCC     2021-2022         ART 115         ART 115         ART 115         ART 115         1.0      1                       512.0          0.826470
# 1       PCC     2021-2022        ART 140A         ART 263        ART 140A            None         NaN     -1                         NaN               NaN
# 2       PCC     2020-2021         ART 273         ART 271         ART 273            None         NaN     -1                         NaN               NaN
# 3       PCC     2019-2020         ATH 101        ANTH 240         ATH 101        ANTH 240         1.0      1                       382.0          0.724287
# 4       PCC     2021-2022          BA 211          BA 211          BA 211          BA 211         1.0      1                       770.0          0.778878
# 5       PCC     2019-2020          ES 101          ES 101          ES 101          ES 101         1.0      1                      2523.0          0.910678
# 6       PCC     2019-2020           G 201         GEO 201           G 201         GEO 201         1.0      1                      3099.0          0.732851
# 7       PCC     2019-2020           G 202         GEO 202           G 202         GEO 202         1.0      1                      3101.0          0.718366
# 8       PCC     2019-2020           G 203         GEO 203           G 203         GEO 203         1.0      1                      3103.0          0.739311
# 9       PCC     2020-2021         GEO 266        GEOG 362         GEO 266        GEOG 362         5.0      1                      3208.0          0.693846
# 10      PCC     2022-2023         GEO 266        GEOG 460         GEO 266        GEOG 460         6.0      1                      3229.0          0.693444
# 11      PCC     2021-2022         HST 102         HST 102         HST 102         HST 102         1.0      1                      3633.0          0.819753
# 12      PCC     2022-2023         HST 121         HST 104         HST 121         HST 104         1.0      1                      3635.0          0.812520
# 13      PCC     2021-2022         MTH 111         MTH 111         MTH 111         MTH 111         1.0      1                      4310.0          0.816988
# 14      PCC     2021-2022         MTH 112         MTH 112         MTH 112         MTH 112         1.0      1                      4312.0          0.838664
# 15      PCC     2021-2022         MTH 255         MTH 255         MTH 255            None         NaN     -1                         NaN               NaN
# 16      PCC     2020-2021         MUS 105         MUS 101         MUS 105         MUS 101         1.0      1                      4494.0          0.791413
# 17      PCC     2020-2021         PHY 101          PH 106         PHY 101          PH 106         5.0      1                      5108.0          0.675720
# 18      PCC     2020-2021         PHY 122          PH 206         PHY 122            None         NaN      0                         NaN               NaN
# 19      PCC     2020-2021         PHY 123          PH 207         PHY 123            None         NaN      0                         NaN               NaN
# 20      PCC     2020-2021        PSY 201A         PSY 201        PSY 201A            None         NaN     -1                         NaN               NaN
# 21      PCC     2020-2021        PSY 202A         PSY 202        PSY 202A            None         NaN     -1                         NaN               NaN
# 22      PCC     2021-2022         SOC 204         SOC 204         SOC 204         SOC 204         1.0      1                      5799.0          0.878414
# 23      PCC     2020-2021          WR 121          WR 121          WR 121          WR 121         1.0      1                      6324.0          0.804296
# 24      PSU     2019-2020          BA 325          BA 370          BA 325          BA 370         1.0      1                       828.0          0.666495
# 25      PSU     2020-2021          BI 427          BI 454          BI 427          BI 454         1.0      1                      1134.0          0.774808
# 26      PSU     2022-2023          CS 314          CS 361          CS 314            None         NaN      0                         NaN               NaN
# 27      PSU     2019-2020         DES 121          GD 226         DES 121          GD 226         2.0      1                      3068.0          0.652217
# 28      PSU     2019-2020         DES 210         ART 121         DES 210         ART 121         4.0      1                       514.0          0.625234
# 29      PSU     2019-2020         DES 290          GD 269         DES 290          GD 269         1.0      1                      3071.0          0.699090
# 30      PSU     2020-2021        GEOG 230        GEOG 103        GEOG 230        GEOG 103         2.0      1                      3183.0          0.769489
# 31      PSU     2019-2020         MTH 255         MTH 255         MTH 255         MTH 255         2.0      1                      4325.0          0.776020
# 32      PSU     2020-2021         PSY 204         PSY 202         PSY 204         PSY 202         1.0      1                      5405.0          0.756944















udat1 = get_upper_lower_dat(institutions_dict)
udat2 = get_art_tables(institutions_dict)
udat = pd.concat([udat1, udat2]).reset_index(drop=True)





proc_acc(udat, False, False)
proc_acc(udat, True, False)
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





