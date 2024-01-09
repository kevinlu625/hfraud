import pandas as pd

column_data_types = {'Rndrng_Prvdr_State_Abrvtn': str, 'Rndrng_Prvdr_State_FIPS': int}
partBData2021PD = pd.read_csv('/Users/kevinlu/Desktop/medicare data/Medicare_Physician_Other_Practitioners_by_Provider_and_Service_2021.csv',
                              dtype=column_data_types)



