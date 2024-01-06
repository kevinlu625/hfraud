import pandas as pd
from functools import reduce
import numpy as np

# column_data_types = {'Rndrng_Prvdr_State_Abrvtn': str, 'Rndrng_Prvdr_State_FIPS': int}
# partBData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Physician_Other_Practitioners_by_Provider_and_Service_2021.csv',
#                               dtype=column_data_types)
# partDData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Part_D_Prescribers_by_Provider_and_Drug_2021.csv',
#                               dtype={'Prscrbr_State_Abrvtn': str})

# partDData2021PD['Tot_Benes'].fillna(5, inplace=True)
# partDData2021PD['Tot_Suplr_Benes'].fillna(5, inplace=True)

# partBData2021PD['Year'] = 2021
# partDData2021PD['Year'] = 2021

def cleaningDMEForAYear (dmeCSVForYear, dmeProvCSVForYear, year):
    dmeCSVForYear = pd.read_csv(dmeCSVForYear,
                            dtype={'Rfrg_Prvdr_State_Abrvtn': str})
    
    dmeCSVForYear['Tot_Suplr_Benes'].fillna(5, inplace=True)
    dmeCSVForYear['Tot_Suplr_Clms'].fillna(5, inplace=True)

    dmeCSVForYear['Year'] = year

    dmeCSVForYearPrvData = dmeCSVForYear[['Rfrg_NPI', 'Rfrg_Prvdr_Last_Name_Org', 'Rfrg_Prvdr_First_Name', 'Rfrg_Prvdr_MI', 
                                  'Rfrg_Prvdr_Crdntls', 'Rfrg_Prvdr_St1', 'Rfrg_Prvdr_St2', 'Rfrg_Prvdr_City',
                                  'Rfrg_Prvdr_State_Abrvtn', 'Rfrg_Prvdr_Zip5', 'Rfrg_Prvdr_Cntry', 'Rfrg_Prvdr_Type']]
    dmeCSVForYearPrvData.drop_duplicates(inplace=True)

    dmeSummCSVForYear = dmeCSVForYear.groupby('Rfrg_NPI').agg({
        'HCPCS_CD': lambda x: list(set(x)),
        'HCPCS_Desc': lambda x: list(set(x)),
        'BETOS_Lvl': lambda x: list(set(x)),
        'BETOS_Cd': lambda x: list(set(x)),
        'BETOS_Desc': lambda x: list(set(x))
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
    
    dmeSummCSVForYear.fillna(np.nan, inplace=True)

    dmeSummCSVForYear = pd.merge(dmeSummCSVForYear, dmeCSVForYearPrvData, on='Rfrg_NPI', how='inner')

    column_data_types = {'Rndrng_Prvdr_State_Abrvtn': 'str', 'Rndrng_Prvdr_State_FIPS': 'int'}
    dmeProvCSVForYear = pd.read_csv(dmeProvCSVForYear, dtype=column_data_types)
    dmeProvCSVForYear.drop(['Bene_Race_Wht_Cnt', 'Bene_Race_Black_Cnt', 'Bene_Race_Api_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_Natind_Cnt', 'Bene_Race_Othr_Cnt'], axis=1, inplace=True)
    dmeProvCSVForYear.fillna(np.nan, inplace=True)

    cols_to_use = dmeProvCSVForYear.columns.difference(dmeCSVForYear.columns)
    cols_to_use = cols_to_use.tolist()
    cols_to_use.append('Rfrg_NPI')

    dmeDataMerged = pd.merge(dmeSummCSVForYear, dmeProvCSVForYear[cols_to_use], on='Rfrg_NPI')

    # Construct the filename with the year parameter
    filename = f'dmeData{year}Merged.csv'

    # Save the merged DataFrame to a CSV file
    dmeDataMerged.to_csv(filename, index=False)

# cleaningDMEForAYear('/Users/kevinlu/Documents/GitHub/hfraud/data/medicare data/dme2021.csv',
#                     '/Users/kevinlu/Documents/GitHub/hfraud/data/medicare data/dme2021provsumm.csv',
#                     2021)

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

    filename = f'allFraud{year}'

    allFraudForYear.to_csv(filename)

allFraud2021 = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/data/allFraud2021.csv')

dmeData2021Merged = pd.read_csv('/Users/kevinlu/Documents/GitHub/hfraud/dmeData2021Merged.csv')

dmeFinal2021 = pd.merge(dmeData2021Merged, allFraud2021, left_on='Rfrg_NPI', right_on='NPI')

dmeFinal2021.to_csv('dmeFinal2021.csv')