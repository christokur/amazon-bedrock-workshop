import json
import os
import boto3
import random

profile_name = os.getenv("AWS_PROFILE", 'cloud-services-dev')
print(f"{profile_name=}")
region_name = 'us-west-2' # os.getenv("AWS_REGION", 'us-west-2')
print(f"{region_name=}")

boto3_session = boto3.Session(profile_name=profile_name, region_name=region_name)

region_name = boto3_session.region_name
iam_client = boto3_session.client('iam')
account_number = boto3_session.client('sts').get_caller_identity().get('Account')
identity = boto3_session.client('sts').get_caller_identity()['Arn']

suffix = None
encryption_policy_name = None
network_policy_name = None
access_policy_name = None
bedrock_execution_role_name = None
fm_policy_name = None
s3_policy_name = None
oss_policy_name = None


def set_names(suffix_p: str | None = None):
    print("> set_names")
    global suffix, encryption_policy_name, network_policy_name, access_policy_name, \
        bedrock_execution_role_name, fm_policy_name, s3_policy_name, oss_policy_name

    suffix = suffix_p or random.randrange(200, 900)
    encryption_policy_name = f"bedrock-sample-rag-sp-{suffix}"
    print(f"{encryption_policy_name=}")
    network_policy_name = f"bedrock-sample-rag-np-{suffix}"
    print(f"{network_policy_name=}")
    access_policy_name = f'bedrock-sample-rag-ap-{suffix}'
    print(f"{access_policy_name=}")
    bedrock_execution_role_name = f'AmazonBedrockExecutionRoleForKnowledgeBase_{suffix}'
    print(f"{bedrock_execution_role_name=}")
    fm_policy_name = f'AmazonBedrockFoundationModelPolicyForKnowledgeBase_{suffix}'
    print(f"{fm_policy_name=}")
    s3_policy_name = f'AmazonBedrockS3PolicyForKnowledgeBase_{suffix}'
    print(f"{s3_policy_name=}")
    oss_policy_name = f'AmazonBedrockOSSPolicyForKnowledgeBase_{suffix}'
    print(f"{oss_policy_name=}")
    print("< set_names")


def create_bedrock_execution_role(bucket_name):
    print("> create_bedrock_execution_role")
    foundation_model_policy_document = {
        "Version":   "2012-10-17",
        "Statement": [
            {
                "Effect":   "Allow",
                "Action":   [
                    "bedrock:InvokeModel",
                ],
                "Resource": [
                    f"arn:aws:bedrock:{region_name}::foundation-model/amazon.titan-embed-text-v1"
                ]
            }
        ]
    }

    s3_policy_document = {
        "Version":   "2012-10-17",
        "Statement": [
            {
                "Effect":    "Allow",
                "Action":    [
                    "s3:GetObject",
                    "s3:ListBucket"
                ],
                "Resource":  [
                    f"arn:aws:s3:::{bucket_name}",
                    f"arn:aws:s3:::{bucket_name}/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:ResourceAccount": f"{account_number}"
                    }
                }
            }
        ]
    }

    assume_role_policy_document = {
        "Version":   "2012-10-17",
        "Statement": [
            {
                "Effect":    "Allow",
                "Principal": {
                    "Service": "bedrock.amazonaws.com"
                },
                "Action":    "sts:AssumeRole"
            }
        ]
    }
    # create policies based on the policy documents
    print(f"{fm_policy_name=}")
    policies = _get_all_iam_user_policies()
    if fm_policy_name in policies:
        print(f"Foundation model policy already exists")
        fm_policy_arn = policies[fm_policy_name]["Arn"]
    else:
        fm_policy = iam_client.create_policy(
            PolicyName=fm_policy_name,
            PolicyDocument=json.dumps(foundation_model_policy_document),
            Description='Policy for accessing foundation model',
        )
        fm_policy_arn = fm_policy["Policy"]["Arn"]

    print(f"{s3_policy_name=}")
    if s3_policy_name in policies:
        print(f"S3 policy already exists")
        s3_policy_arn = policies[s3_policy_name]["Arn"]
    else:
        s3_policy = iam_client.create_policy(
            PolicyName=s3_policy_name,
            PolicyDocument=json.dumps(s3_policy_document),
            Description='Policy for reading documents from s3')
        s3_policy_arn = s3_policy["Policy"]["Arn"]

    # create bedrock execution role
    print(f"{bedrock_execution_role_name=}")
    roles = _get_all_iam_roles()
    if bedrock_execution_role_name in roles:
        print(f"Bedrock execution role already exists")
        bedrock_kb_execution_role = roles[bedrock_execution_role_name]
    else:
        bedrock_kb_execution_role = iam_client.create_role(
            RoleName=bedrock_execution_role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description='Amazon Bedrock Knowledge Base Execution Role for accessing OSS and S3',
            MaxSessionDuration=3600
        )
        bedrock_kb_execution_role = bedrock_execution_role["Role"]
    bedrock_kb_execution_role_arn = bedrock_kb_execution_role['Arn']

    # attach policies to Amazon Bedrock execution role
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["RoleName"],
        PolicyArn=fm_policy_arn
    )
    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["RoleName"],
        PolicyArn=s3_policy_arn
    )
    return bedrock_kb_execution_role


def _get_all_iam_roles():
    response = iam_client.list_roles(MaxItems=1000)
    roles = {role['RoleName']: role for role in response['Roles']}
    while response.get('IsTruncated', False):
        response = iam_client.list_roles(Marker=response['Marker'])
        roles.update({role['RoleName']: role for role in response['Roles']})
    return roles


def _get_all_iam_user_policies():
    response = iam_client.list_policies(Scope='Local')
    policies = {policy['PolicyName']: policy for policy in response['Policies']}
    while response.get('IsTruncated', False):
        response = iam_client.list_policies(Scope='Local', Marker=response['Marker'])
        policies.update({policy['PolicyName']: policy for policy in response['Policies']})
    return policies


def create_oss_policy_attach_bedrock_execution_role(collection_id, bedrock_kb_execution_role):
    # define oss policy document
    oss_policy_document = {
        "Version":   "2012-10-17",
        "Statement": [
            {
                "Effect":   "Allow",
                "Action":   [
                    "aoss:APIAccessAll"
                ],
                "Resource": [
                    f"arn:aws:aoss:{region_name}:{account_number}:collection/{collection_id}"
                ]
            }
        ]
    }
    policies = _get_all_iam_user_policies()
    if oss_policy_name in policies:
        print(f"Opensearch serverless policy already exists")
        oss_policy_arn = policies[oss_policy_name]["Arn"]
    else:
        oss_policy = iam_client.create_policy(
            PolicyName=oss_policy_name,
            PolicyDocument=json.dumps(oss_policy_document),
            Description='Policy for accessing opensearch serverless',
        )
        oss_policy_arn = oss_policy["Policy"]["Arn"]
    print("Opensearch serverless arn: ", oss_policy_arn)

    iam_client.attach_role_policy(
        RoleName=bedrock_kb_execution_role["RoleName"],
        PolicyArn=oss_policy_arn
    )
    return None


def create_policies_in_oss(vector_store_name, aoss_client, bedrock_kb_execution_role_arn):
    response = aoss_client.list_security_policies(
        type='encryption'
    )
    encryption_policies = {policy['name']: policy for policy in response['securityPolicySummaries']}
    while response.get('nextToken', False):
        response = aoss_client.list_security_policies(
            type='encryption',
            nextToken=response['nextToken']
        )
        encryption_policies.update({policy['name']: policy for policy in response['securityPolicySummaries']})
    if encryption_policy_name in encryption_policies:
        print(f"Encryption policy already exists")
        encryption_policy = encryption_policies[encryption_policy_name]
    else:
        encryption_policy = aoss_client.create_security_policy(
            name=encryption_policy_name,
            policy=json.dumps(
                {
                    'Rules':       [
                        {
                            'Resource':     ['collection/' + vector_store_name],
                            'ResourceType': 'collection',
                        },
                    ],
                    'AWSOwnedKey': True,
                }),
            type='encryption'
        )
        encryption_policy = encryption_policy['securityPolicyDetail']

    response = aoss_client.list_security_policies(
        type='network'
    )
    network_policies = {policy['name']: policy for policy in response['securityPolicySummaries']}
    while response.get('nextToken', False):
        response = aoss_client.list_security_policies(
            type='encryption',
            nextToken=response['nextToken']
        )
        network_policies.update({policy['name']: policy for policy in response['securityPolicySummaries']})
    if network_policy_name in network_policies:
        print(f"Network policy already exists")
        network_policy = network_policies[network_policy_name]
    else:
        network_policy = aoss_client.create_security_policy(
            name=network_policy_name,
            policy=json.dumps(
                [
                    {'Rules':           [{'Resource':     ['collection/' + vector_store_name],
                                          'ResourceType': 'collection'}],
                     'AllowFromPublic': True}
                ]),
            type='network'
        )
        network_policy = network_policy['securityPolicyDetail']
    response = aoss_client.list_access_policies(
        type='data'
    )
    data_policies = {policy['name']: policy for policy in response['accessPolicySummaries']}
    while response.get('nextToken', False):
        response = aoss_client.list_access_policies(
            type='access',
            nextToken=response['nextToken']
        )
        data_policies.update({policy['name']: policy for policy in response['accessPolicySummaries']})
    if access_policy_name in data_policies:
        print(f"Data policy already exists")
        access_policy = data_policies[access_policy_name]
    else:
        access_policy = aoss_client.create_access_policy(
            name=access_policy_name,
            policy=json.dumps(
                [
                    {
                        'Rules':       [
                            {
                                'Resource':     ['collection/' + vector_store_name],
                                'Permission':   [
                                    'aoss:CreateCollectionItems',
                                    'aoss:DeleteCollectionItems',
                                    'aoss:UpdateCollectionItems',
                                    'aoss:DescribeCollectionItems'],
                                'ResourceType': 'collection'
                            },
                            {
                                'Resource':     ['index/' + vector_store_name + '/*'],
                                'Permission':   [
                                    'aoss:CreateIndex',
                                    'aoss:DeleteIndex',
                                    'aoss:UpdateIndex',
                                    'aoss:DescribeIndex',
                                    'aoss:ReadDocument',
                                    'aoss:WriteDocument'],
                                'ResourceType': 'index'
                            }],
                        'Principal':   [identity, bedrock_kb_execution_role_arn],
                        'Description': 'Easy data policy'}
                ]),
            type='data'
        )
        access_policy = access_policy['accessPolicyDetail']
    return encryption_policy, network_policy, access_policy


def delete_iam_role_and_policies(account_number:str, suffix_p: str | None = None):
    set_names(suffix_p)
    assert s3_policy_name, "s3_policy_name"
    s3_policy_arn = f"arn:aws:iam::{account_number}:policy/{s3_policy_name}"
    print(f"{s3_policy_arn=}")
    import contextlib

    assert bedrock_execution_role_name, "bedrock_execution_role_name"
    with contextlib.suppress(Exception):
        iam_client.detach_role_policy(
            RoleName=bedrock_execution_role_name,
            PolicyArn=s3_policy_arn
        )
    assert fm_policy_name, "fm_policy_name"
    fm_policy_arn = f"arn:aws:iam::{account_number}:policy/{fm_policy_name}"
    print(f"{fm_policy_arn=}")
    with contextlib.suppress(Exception):
        iam_client.detach_role_policy(
            RoleName=bedrock_execution_role_name,
            PolicyArn=fm_policy_arn
        )

    assert oss_policy_name, "oss_policy_name"
    oss_policy_arn = f"arn:aws:iam::{account_number}:policy/{oss_policy_name}"
    print(f"{oss_policy_arn=}")
    with contextlib.suppress(Exception):
        iam_client.detach_role_policy(
            RoleName=bedrock_execution_role_name,
            PolicyArn=oss_policy_arn
        )
    
    with contextlib.suppress(Exception):
        print(f"{bedrock_execution_role_name=}")
        iam_client.delete_role(RoleName=bedrock_execution_role_name)
    with contextlib.suppress(Exception):
        print(f"{s3_policy_arn=}")
        iam_client.delete_policy(PolicyArn=s3_policy_arn)
    with contextlib.suppress(Exception):
        print(f"{fm_policy_arn=}")
        iam_client.delete_policy(PolicyArn=fm_policy_arn)
    with contextlib.suppress(Exception):
        print(f"{oss_policy_arn=}")
        iam_client.delete_policy(PolicyArn=oss_policy_arn)
    return 0
