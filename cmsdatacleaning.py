import pandas as pd
from functools import reduce

# column_data_types = {'Rndrng_Prvdr_State_Abrvtn': str, 'Rndrng_Prvdr_State_FIPS': int}
# partBData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Physician_Other_Practitioners_by_Provider_and_Service_2021.csv',
#                               dtype=column_data_types)
# partDData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Part_D_Prescribers_by_Provider_and_Drug_2021.csv',
#                               dtype={'Prscrbr_State_Abrvtn': str})

# partDData2021PD['Tot_Benes'].fillna(5, inplace=True)
# partDData2021PD['Tot_Suplr_Benes'].fillna(5, inplace=True)

# partBData2021PD['Year'] = 2021
# partDData2021PD['Year'] = 2021

def cleaningDMEForAYear (dmeCSVForYear, year):
    dmeData2021PD = pd.read_csv(dmeCSVForYear,
                            dtype={'Rfrg_Prvdr_State_Abrvtn': str})
    
    dmeData2021PD['Tot_Suplr_Benes'].fillna(5, inplace=True)
    dmeData2021PD['Tot_Suplr_Clms'].fillna(5, inplace=True)

    dmeData2021PD['Year'] = year

    #we want HCPCS_Cd(HCPCS_Desc) and BETOS_Cd(BETOS_Desc)
    dmeData2021PDAggregated = dmeData2021PD.groupby('Rfrg_NPI').agg({
        'Rfrg_Prvdr_Last_Name_Org': 'first',
        'Rfrg_Prvdr_First_Name': 'first',
        'Rfrg_Prvdr_MI': 'first',
        'Rfrg_Prvdr_Crdntls': 'first',
        'Rfrg_Prvdr_Gndr': 'first',
        'Rfrg_Prvdr_Ent_Cd': 'first',
        'Rfrg_Prvdr_St1': 'first',
        'Rfrg_Prvdr_St2': 'first',
        'Rfrg_Prvdr_City': 'first',
        'Rfrg_Prvdr_State_Abrvtn': 'first',
        'Rfrg_Prvdr_State_FIPS': 'first',
        'Rfrg_Prvdr_Zip5': 'first',
        'Rfrg_Prvdr_RUCA_CAT': 'first',
        'Rfrg_Prvdr_RUCA': 'first',
        'Rfrg_Prvdr_Cntry': 'first',
        'Rfrg_Prvdr_Type_cd': 'first',
        'Rfrg_Prvdr_Type': 'first',
        'Rfrg_Prvdr_Type_Flag': 'first',
        'HCPCS_CD': lambda x: list(set(x)),
        'HCPCS_Desc': lambda x: list(set(x)),
        'BETOS_Lvl': lambda x: list(set(x)),
        'BETOS_Cd': lambda x: list(set(x)),
        'BETOS_Desc': lambda x: list(set(x)),
        'Tot_Suplrs': ['min', 'max', 'median', 'mean', 'sum'],
        'Tot_Suplr_Benes': ['min', 'max', 'median', 'mean', 'sum'],
        'Tot_Suplr_Clms': ['min', 'max', 'median', 'mean', 'sum'],
        'Tot_Suplr_Srvcs': ['min', 'max', 'median', 'mean', 'sum'],
        'Avg_Suplr_Sbmtd_Chrg': ['min', 'max', 'median', 'mean', 'sum'],
        'Avg_Suplr_Mdcr_Alowd_Amt': ['min', 'max', 'median', 'mean', 'sum'],
        'Avg_Suplr_Mdcr_Pymt_Amt': ['min', 'max', 'median', 'mean', 'sum'],
        'Avg_Suplr_Mdcr_Stdzd_Amt': ['min', 'max', 'median', 'mean', 'sum']
    }).reset_index()

    dmeData2021PDAggregated.columns = ['{}_{}'.format(col[0], col[1]) if col[1] != '' else col[0] for col in dmeData2021PDAggregated.columns]

    column_data_types = {'Rndrng_Prvdr_State_Abrvtn': 'str', 'Rndrng_Prvdr_State_FIPS': 'int'}
    dmeData2021PDSummaryStats = pd.read_csv('/Users/kevinlu/Desktop/medicare data/dme2021provsumm.csv', 
                                            dtype=column_data_types)
    dmeData2021PDSummaryStats.drop(['Bene_Race_Wht_Cnt', 'Bene_Race_Black_Cnt', 'Bene_Race_Api_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_Natind_Cnt', 'Bene_Race_Othr_Cnt'], axis=1, inplace=True)
    dmeData2021PDSummaryStats.fillna(0, inplace=True)

    dmeData2021PDMerged = pd.merge(dmeData2021PDAggregated, dmeData2021PDSummaryStats, on='Rfrg_NPI')

    dmeData2021PDMergedCSV = 'dmeData2021PDMergedCSV.csv'

    dmeData2021PDMerged.to_csv(dmeData2021PDMergedCSV, index=False)
    
#getting all fraudsters for a 2021 year and their corresponding termination time
jan2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2101EXCL.csv')
feb2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2102EXCL.csv')
march2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2103EXCL.csv')
april2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2104EXCL.csv')
may2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2105EXCL.csv')
june2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2106EXCL.csv')
july2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2107EXCL.csv')
august2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2108EXCL.csv')
sep2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2109EXCL.csv')
oct2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2110excl.csv')
nov2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2111excl.csv')
dec2021Fraud = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/2112excl.csv')

# List of all DataFrames
fraud_dfs = [jan2021Fraud, feb2021Fraud, march2021Fraud, april2021Fraud, may2021Fraud, june2021Fraud,
             july2021Fraud, august2021Fraud, sep2021Fraud, oct2021Fraud, nov2021Fraud, dec2021Fraud]

def cleaningAllFraudstersForAYear (janToDecFraudReport):
    # Performing a union of all DataFrames
    merged_fraud_df = pd.concat(janToDecFraudReport, ignore_index=True)

    merged_fraud_df.drop(merged_fraud_df[merged_fraud_df['NPI'] == 0].index, inplace=True)

    merged_fraud_df['Year'] = 2021

    fraudCodeWithTerminationTime = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/fraudCodeWithTerminationTime.csv')


    fraudMergedWithYear = pd.merge(merged_fraud_df, fraudCodeWithTerminationTime, left_on='EXCLTYPE', right_on='Code', how='left')

    fraudMergedWithYear.set_index('NPI', inplace=True)

    fraudMergedWithYear.dropna(axis=1, how='all', inplace=True)

    print(fraudMergedWithYear)

    fraudMergedWithYear.to_csv('fraudMergedWithYear.csv')

# dmeData2021PDMerged = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/dmeData2021PDMergedCSV.csv')
# allFraudsterFor2021 = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/fraudMergedWithYear.csv')

# # allFraudsterFor2021.set_index('NPI', inplace=True)
# allFraudsterFor2021.drop('Unnamed: 0', inplace=True)

# # print(dmeData2021PDMerged.columns)

# print(allFraudsterFor2021.columns)
    
# cleaningAllFraudstersForAYear(fraud_dfs)