import pandas as pd
from functools import reduce
import numpy as np
import json

# column_data_types = {'Rndrng_Prvdr_State_Abrvtn': str, 'Rndrng_Prvdr_State_FIPS': int}
# partBData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Physician_Other_Practitioners_by_Provider_and_Service_2021.csv',
#                               dtype=column_data_types)
# partDData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Part_D_Prescribers_by_Provider_and_Drug_2021.csv',
#                               dtype={'Prscrbr_State_Abrvtn': str})

# partDData2021PD['Tot_Benes'].fillna(5, inplace=True)
# partDData2021PD['Tot_Suplr_Benes'].fillna(5, inplace=True)

# partBData2021PD['Year'] = 2021
# partDData2021PD['Year'] = 2021

def cleaningDMEForAYear (dmeProvAndServCSVForYear, dmeProvCSVForYear, year):
    dmeProvAndServCSVForYear = pd.read_csv(dmeProvAndServCSVForYear,
                            dtype={'Rfrg_Prvdr_State_Abrvtn': str})

    dmeProvAndServCSVForYear = dmeProvAndServCSVForYear[['Rfrg_NPI', 'Rfrg_Prvdr_Last_Name_Org', 'Rfrg_Prvdr_First_Name', 'Rfrg_Prvdr_MI', 
                                  'Rfrg_Prvdr_Crdntls','Rfrg_Prvdr_Type', 'HCPCS_CD']]
    dmeProvAndServCSVForYear.drop_duplicates(inplace=True)

    dmeSummCSVForYear = dmeProvAndServCSVForYear.groupby('Rfrg_NPI').agg({
        'HCPCS_CD': lambda x: list(set(x)),
        # 'HCPCS_Desc': lambda x: list(set(x)),
        # 'BETOS_Lvl': lambda x: list(set(x)),
        # 'BETOS_Cd': lambda x: list(set(x)),
        # 'BETOS_Desc': lambda x: list(set(x))
        # 'Tot_Suplrs': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Tot_Suplr_Benes': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Tot_Suplr_Clms': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Tot_Suplr_Srvcs': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Avg_Suplr_Sbmtd_Chrg': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Avg_Suplr_Mdcr_Alowd_Amt': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Avg_Suplr_Mdcr_Pymt_Amt': ['min', 'max', 'median', 'mean', 'sum'],
        # 'Avg_Suplr_Mdcr_Stdzd_Amt': ['min', 'max', 'median', 'mean', 'sum']
    }).reset_index()

    #flattening all columns out
    # dmeSummCSVForYear.columns = ['{}_{}'.format(col[0], col[1]) if col[1] != '' else col[0] for col in dmeCSVForYear.columns]
    
    dmeSummCSVForYear = pd.merge(dmeProvAndServCSVForYear, dmeSummCSVForYear, on='Rfrg_NPI', how='inner')

    dmeSummCSVForYear.fillna(np.nan, inplace=True)

    dmeProvCSVForYear = pd.read_csv(dmeProvCSVForYear)
    dmeProvCSVForYear.drop(['Rfrg_Prvdr_Gnder','Rfrg_Prvdr_Ent_Cd', 'Rfrg_Prvdr_St1', 'Bene_Age_65_74_Cnt', 'Bene_Age_75_84_Cnt',
       'Bene_Age_GT_84_Cnt', 'Bene_Age_LT_65_Cnt','Bene_Race_Wht_Cnt', 
       'Bene_Race_Black_Cnt', 'Bene_Race_Api_Cnt', 'Bene_Race_Hspnc_Cnt', 
       'Bene_Race_Natind_Cnt', 'Bene_Race_Othr_Cnt', 'Bene_Dual_Cnt', 'Bene_Feml_Cnt',
       'Bene_Male_Cnt', 'Bene_Ndual_Cnt', 'DME_Sprsn_Ind', 'DME_Suplr_Mdcr_Pymt_Amt', 'Drug_Sprsn_Ind',
       'Drug_Suplr_Mdcr_Pymt_Amt', 'POS_Sprsn_Ind', 'POS_Suplr_Mdcr_Pymt_Amt', 'Suplr_Mdcr_Alowd_Amt',
       'Suplr_Mdcr_Pymt_Amt', 'Suplr_Mdcr_Stdzd_Pymt_Amt', 'Suplr_Sbmtd_Chrgs',
       'Tot_Suplr_HCPCS_Cds',], axis=1, inplace=True)

    cols_to_use = dmeProvCSVForYear.columns.difference(dmeProvAndServCSVForYear.columns)
    cols_to_use = cols_to_use.tolist()
    cols_to_use.append('Rfrg_NPI')

    dmeDataMerged = pd.merge(dmeSummCSVForYear, dmeProvCSVForYear[cols_to_use], on='Rfrg_NPI')

    dmeDataMerged.fillna(pd.NA, inplace=True)

    dmeDataMerged['Year'] = year

    # Construct the filename with the year parameter
    filename = f'dmeData{year}Merged.csv'

    # Save the merged DataFrame to a CSV file
    dmeDataMerged.to_csv(filename, index=False)

# cleaningDMEForAYear('/Users/kevinlu/Documents/GitHub/hfraud/data/medicare data/dme2021.csv',
#                     '/Users/kevinlu/Documents/GitHub/hfraud/data/medicare data/dme2021provsumm.csv',
#                     2021)

#getting all fraudsters for a 2021 year and their corresponding termination time
# jan2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2101EXCL.csv')
# feb2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2102EXCL.csv')
# march2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2103EXCL.csv')
# april2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2104EXCL.csv')
# may2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2105EXCL.csv')
# june2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2106EXCL.csv')
# july2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2107EXCL.csv')
# august2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2108EXCL.csv')
# sep2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2109EXCL.csv')
# oct2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2110excl.csv')
# nov2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2111excl.csv')
# dec2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2112excl.csv')

# List of all DataFrames
# fraud_dfs = [jan2021Fraud, feb2021Fraud, march2021Fraud, april2021Fraud, may2021Fraud, june2021Fraud,
#              july2021Fraud, august2021Fraud, sep2021Fraud, oct2021Fraud, nov2021Fraud, dec2021Fraud]

def cleaningAllFraudstersForAYear (janToDecFraudReport, year):
    # Performing a union of all DataFrames
    allFraudForYear = pd.concat(janToDecFraudReport, ignore_index=True)

    allFraudForYear.drop(allFraudForYear[allFraudForYear['NPI'] == 0].index, inplace=True)

    allFraudForYear['Year'] = 2021

    fraudCodeWithTerminationTime = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/fraudCodeWithTerminationTime.csv')

    allFraudForYear = pd.merge(allFraudForYear, fraudCodeWithTerminationTime, left_on='EXCLTYPE', right_on='Code', how='left')

    allFraudForYear.set_index('NPI', inplace=True)

    allFraudForYear.dropna(axis=1, how='all', inplace=True)

    allFraudForYear.drop(['REINDATE', 'WAIVERDATE', 'LASTNAME', 'FIRSTNAME', 'MIDNAME', 'BUSNAME', 'GENERAL', 'SPECIALTY', 'DOB', 'ADDRESS', 'CITY', 'STATE', 'ZIP'], axis=1, inplace=True)

    allFraudForYear = allFraudForYear.loc[:, ~allFraudForYear.columns.str.contains('^Unnamed')]

    allFraudForYear.fillna(pd.NA, inplace=True)

    filename = f'allFraud{year}'

    allFraudForYear.to_csv(filename)

def labelFinalDataset (allFraudForYear, dataset, year):
    allFraudForYear = pd.read_csv(allFraudForYear)

    dmeDataMergedForYear = pd.read_csv(dataset)

    # Merge DataFrames on 'Rfrg_NPI' and use 'how' parameter to specify the type of merge
    merged_df = pd.merge(dmeDataMergedForYear, allFraudForYear, left_on='Rfrg_NPI', right_on='NPI', how='left')

    # Create a new column 'fraud' based on the presence of 'Rfrg_NPI' in the right DataFrame
    merged_df['fraud'] = merged_df['Rfrg_NPI'].isin(allFraudForYear['NPI']).map({True: 'fraudulent', False: 'not fraudulent'})

    filename = f'dmeFinal{year}.csv'

    merged_df.to_csv(filename)

# labelFinalDataset('/Users/kevinlu/Documents/GitHub/hfraud/data/allFraud2021.csv',
#                   '/Users/kevinlu/Documents/GitHub/hfraud/dmeData2021Merged.csv',
#                   2021)

def trainingForGPTJSONL(row):
    sentence = "Please determine whether or not this provider is fraudulent: "
    sentence += f"The provider’s name is {row['Rfrg_Prvdr_First_Name']} {row['Rfrg_Prvdr_MI']} {row['Rfrg_Prvdr_Last_Name_Org']} {row['Rfrg_Prvdr_Crdntls']}."
    sentence += f" They practice {row['Rfrg_Prvdr_Type']}. In {row['Year']}, they charged Medicare for these HCPCS codes: {row['HCPCS_CD']}."
    sentence += f" Their beneficiaries' average age was {row['Bene_Avg_Age']}. Their beneficiaries' average risk score was {row['Bene_Avg_Risk_Scre']}."

    conditions = {
        'Atrial Fibrillation': 'Bene_CC_AF_Pct',
        'Alzheimer’s/Dementia': 'Bene_CC_Alzhmr_Pct',
        'Asthma': 'Bene_CC_Asthma_Pct',
        'Chronic Kidney Disease': 'Bene_CC_CKD_Pct',
        'Chronic Obstructive Pulmonary Disease': 'Bene_CC_COPD_Pct',
        'Cancer': 'Bene_CC_Cncr_Pct',
        'Diabetes': 'Bene_CC_Dbts_Pct',
        'Depression': 'Bene_CC_Dprssn_Pct',
        'Hyperlipidemia': 'Bene_CC_Hyplpdma_Pct',
        'Hypertension': 'Bene_CC_Hyprtnsn_Pct',
        'Ischemic Heart Disease': 'Bene_CC_IHD_Pct',
        'Osteoporosis': 'Bene_CC_Opo_Pct',
        'Rheumatoid Arthritis / Osteoarthritis': 'Bene_CC_RAOA_Pct',
        'Stroke': 'Bene_CC_Strok_Pct',
        'Schizophrenia': 'Bene_CC_Sz_Pct',
    }

    sentence += " Of this providers patients, "

    for condition, column_name in conditions.items():
        if not pd.isna(row[column_name]):
            sentence += f" {row[column_name] * 100}% have {condition},"

    if not pd.isna(row['DME_Suplr_Sbmtd_Chrgs']):
        sentence += f" They ordered a total of ${row['DME_Suplr_Sbmtd_Chrgs']} for all durable medical equipment (DME) products/services."
    if not pd.isna(row['DME_Suplr_Mdcr_Alowd_Amt']):
        sentence += f" Their Medicare allowed amount for all DME products/services was ${row['DME_Suplr_Mdcr_Alowd_Amt']}."
    if not pd.isna(row['DME_Suplr_Mdcr_Stdzd_Pymt_Amt']):
        sentence += f" The standardized amount that Medicare paid them after deductible and coinsurance amounts was ${row['DME_Suplr_Mdcr_Stdzd_Pymt_Amt']}."
    if not pd.isna(row['DME_Tot_Suplr_Benes']):
        sentence += f" The total number of unique beneficiaries associated with DME claims ordered by this provider was {row['DME_Tot_Suplr_Benes']}."
    if not pd.isna(row['DME_Tot_Suplr_Clms']):
        sentence += f" The total number of DME claims ordered by this provider was {row['DME_Tot_Suplr_Clms']}." 
    if not pd.isna(row['DME_Tot_Suplr_HCPCS_Cds']):
        sentence += f" The total number of unique DME HCPCS codes ordered by this provider was {row['DME_Tot_Suplr_HCPCS_Cds']}."
    if not pd.isna(row['DME_Tot_Suplr_Srvcs']):
        sentence += f" The total number of DME products/services ordered by this provider was {row['DME_Tot_Suplr_Srvcs']}."
    if not pd.isna(row['DME_Tot_Suplrs']):
        sentence += f" The total number of DME suppliers this provider worked with was {row['DME_Tot_Suplrs']}."

    if not pd.isna(row['Drug_Suplr_Sbmtd_Chrgs']):
        sentence += f"They ordered a total of ${row['Drug_Suplr_Sbmtd_Chrgs']} for all drug and nutritional products/services."
    if not pd.isna(row['Drug_Suplr_Mdcr_Alowd_Amt']):
        sentence += f" Their Medicare allowed amount for all drug and nutritional products/services ordered by this referring provider is ${row['Drug_Suplr_Mdcr_Alowd_Amt']}."
    if not pd.isna(row['Drug_Suplr_Mdcr_Stdzd_Pymt_Amt']):
        sentence += f" The standardized amount that Medicare paid after deductible and coinsurance amounts was ${row['Drug_Suplr_Mdcr_Stdzd_Pymt_Amt']}."
    if not pd.isna(row['Drug_Tot_Suplr_Benes']):
        sentence += f" The total number of unique beneficiaries associated with drug and nutritional product claims ordered by this provider was {row['Drug_Tot_Suplr_Benes']}."
    if not pd.isna(row['Drug_Tot_Suplr_Clms']):
        sentence += f" The total number of drug and nutritional product claims ordered by this provider was {row['Drug_Tot_Suplr_Clms']}."
    if not pd.isna(row['Drug_Tot_Suplr_HCPCS_Cds']):
        sentence += f" The total number of unique drug and nutritional product HCPCS codes ordered by this provider was {row['Drug_Tot_Suplr_HCPCS_Cds']}."
    if not pd.isna(row['Drug_Tot_Suplr_Srvcs']):
        sentence += f" The total number of drug and nutritional products/services ordered by this provider was {row['Drug_Tot_Suplr_Srvcs']}."
    if not pd.isna(row['Drug_Tot_Suplrs']):
        sentence += f" The total number of drug and nutritional suppliers this provider worked with was {row['Drug_Tot_Suplrs']}."

    if not pd.isna(row['POS_Suplr_Sbmtd_Chrgs']):
        sentence += f"They ordered a total of ${row['POS_Suplr_Sbmtd_Chrgs']} for all prosthetic and orthotic (POS) products/services."
    if not pd.isna(row['POS_Suplr_Mdcr_Alowd_Amt']):
        sentence += f" Their Medicare allowed amount for all POS products/services ordered by this referring provider is ${row['POS_Suplr_Mdcr_Alowd_Amt']}."
    if not pd.isna(row['POS_Suplr_Mdcr_Stdzd_Pymt_Amt']):
        sentence += f" The standardized amount that Medicare paid after deductible and coinsurance amounts was ${row['POS_Suplr_Mdcr_Stdzd_Pymt_Amt']}."
    if not pd.isna(row['POS_Tot_Suplr_Benes']):
        sentence += f" The total number of unique beneficiaries associated with POS claims ordered by this provider was {row['POS_Tot_Suplr_Benes']}."
    if not pd.isna(row['POS_Tot_Suplr_Clms']):
        sentence += f" The total number of POS claims ordered by this provider was {row['POS_Tot_Suplr_Clms']}."
    if not pd.isna(row['POS_Tot_Suplr_HCPCS_Cds']):
        sentence += f" The total number of unique POS HCPCS codes ordered by this provider was {row['POS_Tot_Suplr_HCPCS_Cds']}."
    if not pd.isna(row['POS_Tot_Suplr_Srvcs']):
        sentence += f" The total number of POS products/services ordered by this provider was {row['POS_Tot_Suplr_Srvcs']}."
    if not pd.isna(row['POS_Tot_Suplrs']):
        sentence += f" The total number of POS suppliers this provider worked with was {row['POS_Tot_Suplrs']}."

    return sentence

def get_fraud_status(fraud):
    if fraud == 'fraudulent':
        return "This provider is fraudulent"
    elif fraud == 'not fraudulent':
        return "This provider is not fraudulent"
    else:
        return "Fraud status unknown"

dmeFinal2021 = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/dmeFinal2021.csv')

dmeFinal2021['Year'] = 2021

#splitting into training and test set
# Select 500 rows labeled as nonfraudulent
nonfraudulent_rows = dmeFinal2021[dmeFinal2021['fraud'] == 'not fraudulent'].sample(n=300, random_state=42)

# Select 5 rows labeled as fraudulent
fraudulent_rows = dmeFinal2021[dmeFinal2021['fraud'] == 'fraudulent'].sample(n=3, random_state=42)

# Combine the selected rows
selected_rows = pd.concat([nonfraudulent_rows, fraudulent_rows])

# Assuming df is your DataFrame
# Apply the function to each row and create a new column with the results
result_df = pd.DataFrame()
result_df['metadata'] = selected_rows.apply(trainingForGPTJSONL, axis=1)
result_df['result'] = selected_rows.apply(lambda row: get_fraud_status(row['fraud']), axis=1)

# System role content stays constant
system_content = "Robin is a chatbot that, given some provider data, determines whether or not the provider is fraudulent"
jsonl_list = []

for index, row in result_df.iterrows():
    # Constructing the message structure
    message_structure = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{row['metadata']}"},
            {"role": "assistant", "content": row['result']}
        ]
    }
    jsonl_list.append(message_structure)

with open('dme2021SmallGPTTraining.jsonl', 'w') as jsonl_file:
    for jsonl in jsonl_list:
        json.dump(jsonl, jsonl_file)
        jsonl_file.write('\n')

#validation set json
# Get indices of selected rows
selected_indices = nonfraudulent_rows.index.tolist() + fraudulent_rows.index.tolist()

# Select another 300 rows labeled as nonfraudulent that were not already selected
validation_nonfraudulent_rows = dmeFinal2021[(dmeFinal2021['fraud'] == 'not fraudulent') & (~dmeFinal2021.index.isin(selected_indices))].sample(n=300, random_state=42)

# Select another 3 rows labeled as fraudulent that were not already selected
validation_fraudulent_rows = dmeFinal2021[(dmeFinal2021['fraud'] == 'fraudulent') & (~dmeFinal2021.index.isin(selected_indices))].sample(n=3, random_state=42)

# Combine the selected rows
selected_rows = pd.concat([validation_nonfraudulent_rows, validation_fraudulent_rows])

result_df = pd.DataFrame()
result_df['metadata'] = selected_rows.apply(trainingForGPTJSONL, axis=1)
result_df['result'] = selected_rows.apply(lambda row: get_fraud_status(row['fraud']), axis=1)

# System role content stays constant
system_content = "Robin is a chatbot that, given some provider data, determines whether or not the provider is fraudulent"
jsonl_list = []

for index, row in result_df.iterrows():
    # Constructing the message structure
    message_structure = {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{row['metadata']}"},
            {"role": "assistant", "content": row['result']}
        ]
    }
    jsonl_list.append(message_structure)

with open('TESTdme2021SmallGPT.jsonl', 'w') as jsonl_file:
    for jsonl in jsonl_list:
        json.dump(jsonl, jsonl_file)
        jsonl_file.write('\n')