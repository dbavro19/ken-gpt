import boto3
import datetime
from opensearchpy import OpenSearch
from opensearchpy import RequestsHttpConnection, OpenSearch, AWSV4SignerAuth
import langchain
import json
import pypdf
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

bedrock = boto3.client('bedrock-runtime' , 'us-east-1')

def get_embeddings(bedrock, text):
    body_text = json.dumps({"inputText": text})
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType='application/json'

    response = bedrock.invoke_model(body=body_text, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    return embedding

def index_Doc(client,vectors,text,docType,source):
    
    indexDocument={
        'vectors': vectors,
        'text': text,
        'docType': docType,
        'source': source
        }

    response = client.index(
        index = 'product-docs',
        body = indexDocument,
    #    id = '1', commenting out for now
        refresh = False
    )
    return response


def split_doc(doc):

    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=5000, chunk_overlap=1000
    )

    docs = text_splitter.create_documents([doc])

    return docs



host = '14dzfsbbbt70yuz57f23.us-west-2.aoss.amazonaws.com'
region = 'us-west-2'
service = 'aoss'
credentials = boto3.Session().get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)



###Choose File
#loader = UnstructuredHTMLLoader("ECS_ Appliance_ ECS Guidance on 500 error rate response in ECS _ Dell Ireland.mhtml")
#data = loader.load()
#print(data.page_content)

loader = PyPDFLoader("ecs-space-reclamation-sr-garbage-collection-gc-basic-information-gathering.pdf")
data = loader.load()

#Set metadata manually per doc
docType='KB'
source='https://www.dell.com/support/kbdoc/en-nz/000019395/ecs-space-reclamation-sr-garbage-collection-gc-basic-information-gathering'


for page in data:
    text = page.page_content
    vectors = get_embeddings(bedrock, text)
    response = index_Doc(client,vectors,text,docType,source)




