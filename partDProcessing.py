import pandas as pd

partDData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Part_D_Prescribers_by_Provider_and_Drug_2021.csv',
                              dtype={'Prscrbr_State_Abrvtn': str})

partDData2021PD['Tot_Benes'].fillna(5, inplace=True)
partDData2021PD['Tot_Suplr_Benes'].fillna(5, inplace=True)