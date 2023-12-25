import pandas as pd
import json

beneficiaryData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train_Beneficiarydata-1542865627584.csv')
inpatientData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train_Inpatientdata-1542865627584.csv')
outpatientData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train_Outpatientdata-1542865627584.csv')
labeledData = pd.read_csv('/Users/kevinlu/Desktop/claims data/Train-1542865627584.csv')

inpatientData.drop(['ClmProcedureCode_6', 'ClmProcedureCode_5', 'ClmProcedureCode_4'], axis=1, inplace=True)
outpatientData.drop(['ClmProcedureCode_6', 'ClmProcedureCode_5', 'ClmProcedureCode_4'], axis=1, inplace=True)

inpatientLabeledData = pd.merge(inpatientData, labeledData, on='Provider')

def pdToSentence(df):
    sentence_template = "{Provider} has a medical claim with id {ClaimID} from beneficiary {BeneID} that was started on {ClaimStartDt} and ended on {ClaimEndDt} for a reimbursement amount of {InscClaimAmtReimbursed}. The attending physician for this claim was {AttendingPhysician} and the operating physician was {OperatingPhysician}. The claim diagnosis codes were {ClmDiagnosisCode_1}, {ClmDiagnosisCode_2}, {ClmDiagnosisCode_3}, {ClmDiagnosisCode_4}, {ClmDiagnosisCode_5}, {ClmDiagnosisCode_6}, {ClmDiagnosisCode_7}, {ClmDiagnosisCode_8}, {ClmDiagnosisCode_9}, {ClmDiagnosisCode_10}. The procedure codes were {ClmProcedureCode_1}, {ClmProcedureCode_2}, {ClmProcedureCode_3}."

    def get_fraud_status(fraud):
        if fraud == 'Yes':
            return "This provider is considered to be fraudulent"
        elif fraud == 'No':
            return "This provider is not considered to be fraudulent"
        else:
            return "Fraud status unknown"

    result_df = pd.DataFrame()
    result_df['metadata'] = df.apply(lambda row: sentence_template.format(**row) + " ", axis=1)
    result_df['result'] = df.apply(lambda row: get_fraud_status(row['PotentialFraud']), axis=1)
    return result_df

resultingInpatientStrgData = pdToSentence(inpatientLabeledData)

print(resultingInpatientStrgData)

def pdSentenceToJSON(pd, output_file):
    json_data = [{"text": row} for row in pd['sentence'].tolist()]

    # Save to JSON file
    with open(output_file, 'w') as json_file:
        json.dump(json_data, json_file, indent=1)

    return output_file

# resultingInpatientStrgDataJSON = pdSentenceToJSON(resultingInpatientStrgData, "inpatientJSON.jsonl")

## data checking for gpt training

# data_path = "/Users/kevinlu/Documents/GitHub/hfraud/inpatientJSON.jsonl"

# # Load the dataset
# with open(data_path, 'r') as f:
#     dataset = [json.loads(line) for line in f]
#     print(dataset)

# # Initial dataset stats
# print("Num examples:", len(dataset))
# print("First example:")
# for message in dataset[0]["messages"]:
#     print(message)