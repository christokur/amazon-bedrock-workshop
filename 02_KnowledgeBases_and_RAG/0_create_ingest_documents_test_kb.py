#!python3
from __future__ import annotations

# import contextlib
import warnings
warnings.filterwarnings('ignore')

import boto3
import pathlib
import time
import sys
import json
import os
import pprint
# import random
from retrying import retry
from utility import create_bedrock_execution_role, create_oss_policy_attach_bedrock_execution_role, create_policies_in_oss, set_names
from urllib.request import urlretrieve

profile_name = os.getenv("AWS_PROFILE", 'cloud-services-dev')
region_name = os.getenv("AWS_REGION", 'us-west-2')

boto3_session = boto3.Session(profile_name=profile_name, region_name=region_name)

sts_client = boto3_session.client('sts')
# boto3_session = boto3.session.Session()
region_name = boto3_session.region_name
bedrock_agent_client = boto3_session.client('bedrock-agent', region_name=region_name)
service = 'aoss'
s3_client = boto3_session.client('s3')
account_id = sts_client.get_caller_identity()["Account"]
s3_suffix = f"{region_name}-{account_id}"
bucket_name = f'bedrock-kb-{s3_suffix}'  # replace it with your bucket name.
pp = pprint.PrettyPrinter(indent=2)

# Create S3 bucket for knowledge base data source
response = s3_client.list_buckets()
buckets = [bucket['Name'] for bucket in response['Buckets']]
if bucket_name not in buckets:
    print(f'Creating bucket {bucket_name}')
    s3bucket = s3_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': region_name}
    )
else:
    print(f'Bucket {bucket_name} already exists')

# suffix = random.randrange(200, 900)
suffix = "724"
set_names(suffix)
vector_store_name = f'bedrock-sample-rag-{suffix}'
index_name = f"bedrock-sample-rag-index-{suffix}"
aoss_client = boto3_session.client('opensearchserverless')
bedrock_kb_execution_role = create_bedrock_execution_role(bucket_name=bucket_name)
bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Arn']

# create security, network and data access policies within OSS
encryption_policy, network_policy, access_policy = create_policies_in_oss(
    vector_store_name=vector_store_name,
    aoss_client=aoss_client,
    bedrock_kb_execution_role_arn=bedrock_kb_execution_role_arn,
)
response = aoss_client.list_collections(
    collectionFilters={
        'name': vector_store_name,
        # 'status': 'CREATING'|'DELETING'|'ACTIVE'|'FAILED'
    },
    maxResults=100,
)
if response.get('collectionSummaries', []):
    print(f"Collection {vector_store_name} already exists")
    collection = response['collectionSummaries'][0]
else:
    collection = aoss_client.create_collection(name=vector_store_name, type='VECTORSEARCH')
    collection = collection['createCollectionDetail']

pp.pprint(collection)

collection_id = collection['id']
host = collection_id + '.' + region_name + '.aoss.amazonaws.com'
print(host)

# wait for collection creation
response = aoss_client.batch_get_collection(names=[vector_store_name])
# Periodically check collection status
while (response['collectionDetails'][0]['status']) == 'CREATING':
    print('Creating collection...')
    time.sleep(30)
    response = aoss_client.batch_get_collection(names=[vector_store_name])
print('\nCollection successfully created:')
print(response["collectionDetails"])

# create oss policy and attach it to Bedrock execution role
create_oss_policy_attach_bedrock_execution_role(
    collection_id=collection_id,
    bedrock_kb_execution_role=bedrock_kb_execution_role,
)
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

credentials = boto3.Session().get_credentials()
awsauth = auth = AWSV4SignerAuth(credentials, region_name, service)

index_name = f"bedrock-sample-index-{suffix}"
body_json = {
    "settings": {
        "index.knn":                "true",
        "number_of_shards":         1,
        "knn.algo_param.ef_search": 512,
        "number_of_replicas":       0,
    },
    "mappings": {
        "properties": {
            "vector":        {
                "type":      "knn_vector",
                "dimension": 1536,
                "method":    {
                    "name":   "hnsw",
                    "engine": "faiss"
                },
            },
            "text":          {
                "type": "text"
            },
            "text-metadata": {
                "type": "text"}
        }
    }
}
# Build the OpenSearch client
oss_client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    timeout=300
)

# Create index
if not (existing := oss_client.indices.exists(index=index_name)):
    # # It can take up to a minute for data access rules to be enforced
    time.sleep(60)
    response = oss_client.indices.create(index=index_name, body=json.dumps(body_json))
    print('\nCreating index:')
    print(response)
    time.sleep(60)  # index creation can take up to a minute
else:
    print(f"Index {index_name} already exists")
    # pp.pprint(existing)

# Download and prepare dataset

data_root = "./data/"
dp = pathlib.Path(data_root)
dp.mkdir(parents=False, exist_ok=True)

filenames = [
    'AMZN-2022-Shareholder-Letter.pdf',
    'AMZN-2021-Shareholder-Letter.pdf',
    'AMZN-2020-Shareholder-Letter.pdf',
    'AMZN-2019-Shareholder-Letter.pdf'
]
if not all([dp.joinpath(fn).exists() for fn in filenames]):
    urls = [
    'https://s2.q4cdn.com/299287126/files/doc_financials/2023/ar/2022-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/ar/2021-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/ar/Amazon-2020-Shareholder-Letter-and-1997-Shareholder-Letter.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/ar/2019-Shareholder-Letter.pdf'
    ]

    for idx, url in enumerate(urls):
        file_path = data_root + filenames[idx]
        urlretrieve(url, file_path)

# Upload data to s3
s3_client = boto3.client("s3")


def uploadDirectory(path, bucket_name):
    for root, dirs, files in os.walk(path):
        for file in files:
            s3_client.upload_file(os.path.join(root, file), bucket_name, file)


uploadDirectory(data_root, bucket_name)

opensearchServerlessConfiguration = {
    "collectionArn":   collection['arn'],
    "vectorIndexName": index_name,
    "fieldMapping":    {
        "vectorField":   "vector",
        "textField":     "text",
        "metadataField": "text-metadata"
    }
}

chunkingStrategyConfiguration = {
    "chunkingStrategy":               "FIXED_SIZE",
    "fixedSizeChunkingConfiguration": {
        "maxTokens":         512,
        "overlapPercentage": 20
    }
}

s3Configuration = {
    "bucketArn": f"arn:aws:s3:::{bucket_name}",
    # "inclusionPrefixes":["*.*"] # you can use this if you want to create a KB using data within s3 prefixes.
}

embeddingModelArn = f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1"

# Create a KnowledgeBase
@retry(wait_random_min=1000, wait_random_max=2000, stop_max_attempt_number=7)
def create_knowledge_base_func(
    name:str,
    description:str,
    roleArn:str,
) -> dict:
    response = bedrock_agent_client.list_knowledge_bases(
        maxResults=100,
    )
    if response.get('knowledgeBaseSummaries', []):
        print(f"Knowledge base {name} already exists")
        return response['knowledgeBaseSummaries'][0]
    create_kb_response = bedrock_agent_client.create_knowledge_base(
        name=name,
        description=description,
        roleArn=roleArn,
        knowledgeBaseConfiguration={
            "type":                             "VECTOR",
            "vectorKnowledgeBaseConfiguration": {
                "embeddingModelArn": embeddingModelArn
            }
        },
        storageConfiguration={
            "type":                              "OPENSEARCH_SERVERLESS",
            "opensearchServerlessConfiguration": opensearchServerlessConfiguration
        }
    )
    return create_kb_response["knowledgeBase"]


kb = None
try:
    kb = create_knowledge_base_func(
        name = f"bedrock-sample-knowledge-base-{suffix}",
        description = "Amazon shareholder letter knowledge base.",
        roleArn = bedrock_kb_execution_role_arn,
    )
    pp.pprint(kb)
except Exception as err:
    print(f"{err=}, {type(err)=}")
    sys.exit(1)

# Get KnowledgeBase
get_kb_response = bedrock_agent_client.get_knowledge_base(knowledgeBaseId=kb['knowledgeBaseId'])

# Create a DataSource in KnowledgeBase
response = bedrock_agent_client.list_data_sources(
    knowledgeBaseId=kb['knowledgeBaseId'],
)
if response.get('dataSourceSummaries', []):
    print(f"Data source already exists")
    ds = response['dataSourceSummaries'][0]
else:
    create_ds_response = bedrock_agent_client.create_data_source(
        name = f"bedrock-sample-knowledge-base-{suffix}",
        description = "Amazon shareholder letter knowledge base.",
        knowledgeBaseId=kb['knowledgeBaseId'],
        dataSourceConfiguration={
            "type":            "S3",
            "s3Configuration": s3Configuration
        },
        vectorIngestionConfiguration={
            "chunkingConfiguration": chunkingStrategyConfiguration
        }
    )
    ds = create_ds_response["dataSource"]
pp.pprint(ds)

# Get DataSource
bedrock_agent_client.get_data_source(knowledgeBaseId=kb['knowledgeBaseId'], dataSourceId=ds["dataSourceId"])

# Start an ingestion job
start_job_response = bedrock_agent_client.start_ingestion_job(knowledgeBaseId=kb['knowledgeBaseId'], dataSourceId=ds["dataSourceId"])

job = start_job_response["ingestionJob"]
# pp.pprint(f"{job=}")

# Get job
while (job['status'] != 'COMPLETE'):
    time.sleep(40)
    get_job_response = bedrock_agent_client.get_ingestion_job(
        knowledgeBaseId=kb['knowledgeBaseId'],
        dataSourceId=ds["dataSourceId"],
        ingestionJobId=job["ingestionJobId"]
    )
    job = get_job_response["ingestionJob"]
pp.pprint(f"{job=}")

kb_id = kb["knowledgeBaseId"]
pp.pprint(f"{kb_id=}")
