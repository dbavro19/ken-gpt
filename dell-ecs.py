import boto3
import json
#import botocore
#import os
#import sys
from opensearchpy import OpenSearch
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
#import time
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
#import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
import streamlit as st

from requests_aws4auth import AWS4Auth #testing

st.set_page_config(page_title="Ken-GPT", page_icon=":tada", layout="wide")

#Headers
with st.container():
    st.header("Ken-GPT")
    st.subheader("Ask Questions about your ECS and ObjectScale system")
    #st.title("Ask Questions about your ECS and ObjectScale system")

#
with st.container():
    st.write("---")
    userQuery = st.text_input("Ask a Question")
    #userID = st.text_input("User ID")
    st.write("---")




#App Logic
#Embed, Search, Invoke LLM etc.


# Get Embeddings
def get_embedding(bedrock, userQuery):

    body = json.dumps({"inputText": userQuery})
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    return embedding

#Get KNN Results
def get_knn_results(client, userVectors):
    docType = "KB"

    query = {
        "size": 3,
        "query": {
            "knn": {
                "vectors": {
                    "vector": userVectors, "k": 3
                }
            }
        },
        "_source": False,
        "fields": ["text", "source"],
    }


    response = client.search(
        body=query,
        index='product-docs',
    )

    print(response)

    similaritysearchResponse = ""
    count = 1
    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse = similaritysearchResponse + "<context" + str(count) +">" + str(outputtext) + "</context" + str(count) +">"
        outputsource = i["fields"]["source"]
        similaritysearchResponse = similaritysearchResponse + "<source" + str(count) +">" + str(outputsource) + "</context" + str(count) +">"
        similaritysearchResponse = similaritysearchResponse + "\n"

        print("---------------------------------------------------------")
        print(similaritysearchResponse)
        count = count + 1
    
    return similaritysearchResponse

#Invoke LLM - Bedrock
def invoke_llm(bedrock, userQuery, similaritysearchResponse):

    ##Setup Prompt
    prompt_data = f"""
Human: 

You are an AI assistant that will help people answer technical questions about Dell ECS or Dell ObjectScale products
Answer the provided question to the best of your ability using the information provided in the Context.
Summarize the answer and provide sources and the source link to where the relevant information can be found. Include this at the end of the response
Do not include information that is not relevant to the question.
Only provide information based on the context provided, and do not make assumptions.
Format the output in human readable format, use bullet lists and paragraph indents when applicable
Answer concisely with no preamble
If you are unable to answer accurately, please say so

<user_question>
{userQuery}
</user_question>

{similaritysearchResponse}

Return your output in valid markdown format

Assistant: Based on the context provided: 
"""


    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":1000,
                 "temperature":0,
                 "top_k":250,
                 "top_p":0.5,
                 "stop_sequences":[]
                  }) 
    
    #Run Inference
    modelId = "anthropic.claude-instant-v1"  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    llmOutput=response_body.get('completion')

    print(prompt_data)

    #print(llmOutput)

    return llmOutput

def do_it(userQuery):

    #Setup Clients 
    #bedrock client
    bedrock = boto3.client('bedrock-runtime' , 'us-east-1')

    #OpenSearch CLient
    host = '14dzfsbbbt70yuz57f23.us-west-2.aoss.amazonaws.com' # cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
    region = 'us-west-2'
    service = 'aoss'
    credentials = boto3.Session().get_credentials()
    auth = AWSV4SignerAuth(credentials, region, service)

    client = OpenSearch(
        hosts = [{'host': host, 'port': 443}],
        http_auth = auth,
        use_ssl = True,
        verify_certs = True, #Chaging to False for troubleshooting
        connection_class = RequestsHttpConnection,
        pool_maxsize = 20
    )

    #Prep and Get Embeddings from userQuery

    userVectors = get_embedding(bedrock,userQuery)

    #Get KNN Results - Filtering on userID
    similaritysearchResponse = get_knn_results(client, userVectors)


    #invoke LLM
    llmOutput = invoke_llm(bedrock, userQuery, similaritysearchResponse)
    return llmOutput





##Back to Streamlit
result=st.button("ASK!")
if result:
    st.write(do_it(userQuery))


