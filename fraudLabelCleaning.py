import pandas as pd

# getting all fraudsters for a 2021 year and their corresponding termination time
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

    #filtering for healthcare data
    allFraudForYear.drop(allFraudForYear[allFraudForYear['NPI'] == 0].index, inplace=True)

    allFraudForYear['Year'] = year

    fraudCodeWithTerminationTime = pd.read_csv('/Users/kevinlu/Desktop/medicare data/fraud labels/fraudCodeWithTerminationTime.csv')

    allFraudForYear = pd.merge(allFraudForYear, fraudCodeWithTerminationTime, left_on='EXCLTYPE', right_on='Code', how='left')

    allFraudForYear.set_index('NPI', inplace=True)

    allFraudForYear.dropna(axis=1, how='all', inplace=True)

    allFraudForYear.drop(['REINDATE', 'WAIVERDATE', 'LASTNAME', 'FIRSTNAME', 'MIDNAME', 'BUSNAME', 'GENERAL', 'SPECIALTY', 'DOB', 'ADDRESS', 'CITY', 'STATE', 'ZIP'], axis=1, inplace=True)

    allFraudForYear = allFraudForYear.loc[:, ~allFraudForYear.columns.str.contains('^Unnamed')]

    allFraudForYear.fillna(pd.NA, inplace=True)

    filename = f'allFraud{year}'

    allFraudForYear.to_csv(filename)