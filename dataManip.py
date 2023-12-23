import pandas as pd

beneficiaryData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train_Beneficiarydata-1542865627584.csv')
inpatientData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train_Inpatientdata-1542865627584.csv')
outpatientData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train_Outpatientdata-1542865627584.csv')
labeledData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train-1542865627584.csv')

inpatientData.drop(['ClmProcedureCode_6', 'ClmProcedureCode_5', 'ClmProcedureCode_4'], axis=1, inplace=True)
outpatientData.drop(['ClmProcedureCode_6', 'ClmProcedureCode_5', 'ClmProcedureCode_4'], axis=1, inplace=True)

inpatientDataTest = inpatientData

# inpatientData.to_csv("inpatientdatacleaned")

# Perform an inner join on the specified column
# mergedAll = pd.merge(inpatientData, outpatientData, on="Provider")
# fullData = pd.merge(mergedAll, labeledData, on="Provider")

def csvToSentence (df):
    sentence_template = "{Provider} has a medical claim with id {ClaimID} from beneficiary {BeneID} that was started on {ClaimStartDt} and ended on {ClaimEndDt} for a reimbursement amount of {InscClaimAmtReimbursed}. The attending physician for this claim was {AttendingPhysician} and the operating physician was {OperatingPhysician}. The claim diagnosis codes were {ClmDiagnosisCode_1}, {ClmDiagnosisCode_2}, {ClmDiagnosisCode_3}, {ClmDiagnosisCode_4}, {ClmDiagnosisCode_5}, {ClmDiagnosisCode_6}, {ClmDiagnosisCode_7}, {ClmDiagnosisCode_8}, {ClmDiagnosisCode_9}, {ClmDiagnosisCode_10}. The procedure codes were {ClmProcedureCode_1}, {ClmProcedureCode_2}, {ClmProcedureCode_3}."    
    result_df = pd.DataFrame()
    result_df['sentence'] = df.apply(lambda row: sentence_template.format(**row), axis=1)
    return result_df

resultingInpatientStrgData = csvToSentence(inpatientDataTest)
