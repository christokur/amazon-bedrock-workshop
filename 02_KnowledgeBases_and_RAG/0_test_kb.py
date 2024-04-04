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

pp = pprint.PrettyPrinter(indent=2)

profile_name = os.getenv("AWS_PROFILE", 'cloud-services-dev')
region_name = os.getenv("AWS_REGION", 'us-west-2')

session = boto3.Session(profile_name=profile_name, region_name=region_name)
bedrock_agent_client = session.client('bedrock-agent')

# try out KB using RetrieveAndGenerate API
bedrock_agent_runtime_client = session.client("bedrock-agent-runtime")
model_id = "anthropic.claude-instant-v1" # try with both claude instant as well as claude-v2. for claude v2 - "anthropic.claude-v2"
model_arn = f'arn:aws:bedrock:{region_name}::foundation-model/{model_id}'

# Query the knowledge base
kb_id='35ZBLUQRA4'
query = "What is Amazon's doing in the field of generative AI?"
response = bedrock_agent_runtime_client.retrieve_and_generate(
    input={
        'text': query
    },
    retrieveAndGenerateConfiguration={
        'type': 'KNOWLEDGE_BASE',
        'knowledgeBaseConfiguration': {
            'knowledgeBaseId': kb_id,
            'modelArn': model_arn
        }
    },
)

generated_text = response['output']['text']
print(f"{generated_text=}\n")

## print out the source attribution/citations from the original documents to see if the response generated belongs to the context.
citations = response["citations"]
# contexts = []
for citation in citations:
    retrievedReferences = citation["retrievedReferences"]
    for reference in retrievedReferences:
        print(f'{reference["location"]["s3Location"]["uri"]}: {reference["content"]["text"]}')

