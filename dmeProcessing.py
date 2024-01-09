import pandas as pd
from functools import reduce
import numpy as np
import json
import imblearn
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

#292299 unique providers in dmePS2021 dataset, 384264 unique providers in dmeP2021 dataset

def cleaningDMEForAYear (dmeProvAndServCSVForYear, dmeProvCSVForYear, year):
    dmeProvAndServCSVForYear = pd.read_csv(dmeProvAndServCSVForYear,
                            dtype={'Rfrg_Prvdr_State_Abrvtn': str})

    dmeProvAndServCSVForYear = dmeProvAndServCSVForYear.groupby('Rfrg_NPI').agg({
        'HCPCS_CD': lambda x: list(set(x)),
    }).reset_index()

    dmeProvCSVForYear = pd.read_csv(dmeProvCSVForYear)
    dmeProvCSVForYear.drop(['Rfrg_Prvdr_Gndr','Rfrg_Prvdr_Ent_Cd', 'Rfrg_Prvdr_St1', 'Rfrg_Prvdr_St2', 
                            'Bene_Age_65_74_Cnt', 'Bene_Age_75_84_Cnt', 'Rfrg_Prvdr_City', 'Rfrg_Prvdr_State_Abrvtn', 
                            'Rfrg_Prvdr_State_FIPS', 'Rfrg_Prvdr_Zip5', 'Rfrg_Prvdr_RUCA', 'Rfrg_Prvdr_RUCA_Desc',
                            'Rfrg_Prvdr_Cntry', 'Rfrg_Prvdr_Type_Flag','Tot_Suplrs', 'Tot_Suplr_Benes', 'Tot_Suplr_Clms', 'Tot_Suplr_Srvcs',
                            'Bene_Age_GT_84_Cnt', 'Bene_Age_LT_65_Cnt','Bene_Race_Wht_Cnt', 
                            'Bene_Race_Black_Cnt', 'Bene_Race_Api_Cnt', 'Bene_Race_Hspnc_Cnt', 
                            'Bene_Race_Natind_Cnt', 'Bene_Race_Othr_Cnt', 'Bene_Dual_Cnt', 'Bene_Feml_Cnt',
                            'Bene_Male_Cnt', 'Bene_Ndual_Cnt', 'DME_Sprsn_Ind', 'DME_Suplr_Mdcr_Stdzd_Pymt_Amt', 'Drug_Sprsn_Ind',
                            'Drug_Suplr_Mdcr_Stdzd_Pymt_Amt', 'POS_Sprsn_Ind', 'POS_Suplr_Mdcr_Stdzd_Pymt_Amt', 'Suplr_Mdcr_Alowd_Amt',
                            'Suplr_Mdcr_Pymt_Amt', 'Suplr_Mdcr_Stdzd_Pymt_Amt', 'Suplr_Sbmtd_Chrgs',
                            'Tot_Suplr_HCPCS_Cds'], axis=1, inplace=True)
    
    dmeMerged = pd.merge(dmeProvCSVForYear, dmeProvAndServCSVForYear, on='Rfrg_NPI')

    dmeMerged.fillna(pd.NA, inplace=True)

    dmeMerged['Year'] = year

    # Construct the filename with the year parameter
    filename = f'dme{year}Merged.csv'

    # Save the merged DataFrame to a CSV file
    dmeMerged.to_csv(filename, index=False)

# cleaningDMEForAYear('/Users/kevinlu/Documents/GitHub/hfraud/data/medicare data/dmePS2019.csv',
#                     '/Users/kevinlu/Documents/GitHub/hfraud/data/medicare data/dmeP2019.csv', 2019)

def labelFinalDataset (allFraudForYear, dataset, year):
    allFraudForYear = pd.read_csv(allFraudForYear)

    dmeMerged = pd.read_csv(dataset)

    # Create a new column 'fraud' based on the presence of 'Rfrg_NPI' in the right DataFrame
    dmeMerged['fraud'] = dmeMerged['Rfrg_NPI'].isin(allFraudForYear['NPI']).map({True: 'fraudulent', False: 'not fraudulent'})

    filename = f'dmeFinal{year}.csv'

    dmeMerged.to_csv(filename, index=False)

# labelFinalDataset('/Users/kevinlu/Documents/GitHub/hfraud/data/allFraud2019.csv',
#                   '/Users/kevinlu/Documents/GitHub/hfraud/data/dme2019Merged.csv',
#                   2019)

# #ROS + RUS for training and validation to counter class imbalance
dmeFinal2019 = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/data/dmeFinal2019.csv')

X = dmeFinal2019.drop('fraud', axis=1)
y = dmeFinal2019['fraud']

# Split the data into training and test sets (adjust the test_size parameter as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dmeFinal2019TrainSet = pd.concat([X_train, y_train], axis=1)
dmeFinal2019TestSet = pd.concat([X_test, y_test], axis=1)

def oversamplingAndReduction(df, minorityRatio, percentageToKeep):
    #Oversampling
    X = df.drop('fraud', axis=1)
    y = df['fraud']

    # Initialize the RandomOverSampler with the desired sampling strategies
    oversampler = RandomOverSampler(sampling_strategy=minorityRatio, random_state=42)

    X_resampled, y_resampled = oversampler.fit_resample(X, y)

    # Create a new DataFrame with the resampled data
    dmeFinal2021Resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='fraud')], axis=1)

    #Reduction
    dmeFinal2021Reduced = dmeFinal2021Resampled.groupby('fraud', group_keys=False).apply(lambda x: x.sample(frac=percentageToKeep, random_state=42))
    dmeFinal2021Reduced = dmeFinal2021Reduced.sample(frac=1, random_state=42).reset_index(drop=True)

    return dmeFinal2021Reduced

dmeFinal2019TrainSetReduced = oversamplingAndReduction(dmeFinal2019TrainSet, .3, .01)
dmeFinal2019TestSetReduced = oversamplingAndReduction(dmeFinal2019TestSet, .3, .01)

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
    if not pd.isna(row['DME_Suplr_Mdcr_Pymt_Amt']):
        sentence += f" The amount that Medicare paid them after deductible and coinsurance amounts was ${row['DME_Suplr_Mdcr_Pymt_Amt']}."
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
    if not pd.isna(row['Drug_Suplr_Mdcr_Pymt_Amt']):
        sentence += f" The amount that Medicare paid after deductible and coinsurance amounts was ${row['Drug_Suplr_Mdcr_Pymt_Amt']}."
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
    if not pd.isna(row['POS_Suplr_Mdcr_Pymt_Amt']):
        sentence += f" The amount that Medicare paid after deductible and coinsurance amounts was ${row['POS_Suplr_Mdcr_Pymt_Amt']}."
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

# Assuming df is your DataFrame
# Apply the function to each row and create a new column with the results
result_df = pd.DataFrame()
result_df['metadata'] = dmeFinal2019TrainSetReduced.apply(trainingForGPTJSONL, axis=1)
result_df['result'] = dmeFinal2019TrainSetReduced.apply(lambda row: get_fraud_status(row['fraud']), axis=1)

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

with open('dme2019SmallGPTTraining.jsonl', 'w') as jsonl_file:
    for jsonl in jsonl_list:
        json.dump(jsonl, jsonl_file)
        jsonl_file.write('\n')

result_df = pd.DataFrame()
result_df['metadata'] = dmeFinal2019TestSetReduced.apply(trainingForGPTJSONL, axis=1)
result_df['result'] = dmeFinal2019TestSetReduced.apply(lambda row: get_fraud_status(row['fraud']), axis=1)

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

with open('TESTdme2019SmallGPT.jsonl', 'w') as jsonl_file:
    for jsonl in jsonl_list:
        json.dump(jsonl, jsonl_file)
        jsonl_file.write('\n')