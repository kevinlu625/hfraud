import pandas as pd
import requests
import json
import time
from bs4 import BeautifulSoup

def pdToSentenceChat(df):
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

def pdToSentenceText(df):
    sentence_template = "{Provider} has a medical claim with id {ClaimID} from beneficiary {BeneID} that was started on {ClaimStartDt} and ended on {ClaimEndDt} for a reimbursement amount of {InscClaimAmtReimbursed}. The attending physician for this claim was {AttendingPhysician} and the operating physician was {OperatingPhysician}. The claim diagnosis codes were {ClmDiagnosisCode_1}, {ClmDiagnosisCode_2}, {ClmDiagnosisCode_3}, {ClmDiagnosisCode_4}, {ClmDiagnosisCode_5}, {ClmDiagnosisCode_6}, {ClmDiagnosisCode_7}, {ClmDiagnosisCode_8}, {ClmDiagnosisCode_9}, {ClmDiagnosisCode_10}. The procedure codes were {ClmProcedureCode_1}, {ClmProcedureCode_2}, {ClmProcedureCode_3}."

    def get_fraud_status(fraud):
        if fraud == 'Yes':
            return "This provider is considered to be fraudulent"
        elif fraud == 'No':
            return "This provider is not considered to be fraudulent"
        else:
            return "Fraud status unknown"

    result_df = pd.DataFrame()
    result_df['sentence'] = df.apply(lambda row: sentence_template.format(**row) + " " + get_fraud_status(row['PotentialFraud']), axis=1)
    return result_df

def dojPressReleasePDToText(df):
    sentence_template = "{title}.{body}.{date}"

    dojPressReleaseText = pd.DataFrame()
    dojPressReleaseText['data'] = df.apply(lambda row: sentence_template.format(**row), axis=1)
    pdSentenceToJSON(dojPressReleaseText, '')

    return dojPressReleaseText

def pdSentenceToJSON(pd, output_file):
    json_data = [{"text": row} for row in pd['sentence'].tolist()]

    with open(output_file, 'w') as json_file:
        for entry in json_data:
            json_file.write(json.dumps(entry) + '\n')

    return output_file

url = "http://www.justice.gov/api/v1/press_releases.json"
def getDOJPressReleaseInfo(apiurl):
    # resultingInpatientStrgDataJSON = pdSentenceToJSONLlama(resultingInpatientStrgDataTrimmed, "inpatientJSONtrimmed.jsonl")
    url = apiurl
    page = 1
    pagesize = 50
    fields = "title,body,date,topic"
    filtered_data = []

    while True:
        params = {
            "page": page,
            "pagesize": pagesize,
            "fields": fields,
        }

        start_time = time.time()  # Record the start time before making the request

        response = requests.get(url, params=params)

        elapsed_time = time.time() - start_time  # Calculate the time taken for the request

        if response.status_code == 200:
            data = response.json()

            if not data["results"]:
                break  # Break the loop if there are no more results

            df = pd.DataFrame(data["results"], columns=["title", "body", "date", "topic"])
            filtered_df = df[df['topic'].apply(lambda topics: any(topic['name'] == 'Health Care Fraud' for topic in topics))]
            filtered_data.append(filtered_df)

            page += 1  # Move to the next page
            print(page)
            # Introduce a delay to ensure no more than 10 requests per second
            if elapsed_time < 0.1:
                time.sleep(0.1 - elapsed_time)
        else:
            print(f"Failed to retrieve press releases. Status Code: {response.status_code}")
            break  # Break the loop if there's an error

    # Concatenate all DataFrames from different pages into a single DataFrame
    if filtered_data:
        final_df = pd.concat(filtered_data, ignore_index=True)
        final_df.to_csv('dojPressRelease.csv')
        print(final_df)
    else:
        print("No press releases found.")

def fullDOJPressReleaseCleanUp(file):

    dojpressreleasepd = pd.read_csv(file)

    def html_to_text_with_paragraphs(html):
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract text content for each paragraph
        paragraphs = soup.find_all('p')
        paragraph_texts = [paragraph.get_text(separator=' ', strip=True) for paragraph in paragraphs]
        
        # Concatenate paragraph texts
        result_text = '\n\n'.join(paragraph_texts)
        
        return result_text

    # Apply the function to the 'body' column
    dojpressreleasepd['body'] = dojpressreleasepd['body'].apply(html_to_text_with_paragraphs)

    #more cleanup
    dojpressreleasepd.drop_duplicates(subset=['title', 'body', 'date'], inplace=True)
    dojpressreleasepd = dojpressreleasepd.loc[:, ~dojpressreleasepd.columns.str.contains('^Unnamed')]
    dojpressreleasepd.to_csv('/Users/kevinlu/Documents/GitHub/hfraud/data/dojPressRelease.csv')