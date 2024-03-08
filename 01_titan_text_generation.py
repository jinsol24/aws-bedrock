import json

# JSON 데이터를 예쁘게 출력하는 함수수
def printJson(dataObj):
    print(json.dumps(dataObj, sort_keys=True, indent=4)) # JSON 객체를 설정하여 들여쓰기를 포함해 출력


import boto3 # AWS 서비스를 이용하기 위한 boto3 라이브러리 임포트 

# access key id와 secret key가 필요
# bedrock = boto3.client(aws_access_key_id='INPUT YOUR KEY',
#                       aws_secret_access_key='INPUT YOUR KEY')

# Cloud9 EC2는 credential이 설정되어 있으므로 생략 가능
# bedrock을 지원하는 리전은 일부 + 각 모델별 access 요청 필요
bedrock = boto3.client(service_name="bedrock", region_name="us-east-1") # AWS Bedrock 서비스 클라이언트 생성, 저장된 리전에서 Bedrock 서비스 접근 

# list all FMs
# fms = bedrock.list_foundation_models()
# printJson(fms)

# 특정 파운데이션 모델에 대한 정보를 가져옴. 여기서는 'amazon.titan-tq1-large' 모델
titan = bedrock.get_foundation_model(modelIdentifier="amazon.titan-tg1-large")
# printJson(titan)
# 참고: Reference '[201] 모델 예시' 참고

# Bedrock runtime
# boto3.client(service_name='bedrock'): bedrock에 대한 정보
# bedrock을 통해 모델을 실행하려면 'bedrock-runtime' 서비스를 이용
bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1") # Bedrock Runtime 서비스 클라이언트 생성, 모델 실행을 위해 사용 

# 질문을 모델에 입력으로 제공하기 위한 JSON 바디 생성 
question = "What is the AWS Bedrock?"
body = json.dumps({"inputText": question}) # 장문 텍스트를 JSON 형식으로 인코딩 

# 응답 바디에서 결과를 추출하고 JSON으로 파싱 
response = bedrock_runtime.invoke_model(body=body, modelId="amazon.titan-tg1-large")
# print(response)
# 'body': <botocore.response.StreamingBody object at 0x7f067f91c7f0>
response_body = json.loads(response.get("body").read())
# printJson(response_body)
# print(response_body['results'][0]['outputText'])

# 모델 실행에 사용할 세부 설정을 포함한 복잡한 JSON 바디 예시, 텍스트 생성 설정 포함 
# 여기서는 온도(creavity control), topP(filtering parameter), 최대 토큰 수를 설정
# 모델별 적절한 파라미터 전달
# 참고: Reference '[202] Bedrock runtime 파라미터' 참고
body = json.dumps(
    {
        "inputText": question,
        "textGenerationConfig": {
            "temperature": 0.5,
            "topP": 0.5,
            "maxTokenCount": 512,
            "stopSequences": ["something"],
        },
    }
)


# temperature and top p
body = json.dumps(
    {
        "inputText": question,
        "textGenerationConfig": {
            "temperature": 0,
            "topP": 0.01,
            "maxTokenCount": 512,
        },
    }
)

#설정된 파리미터를 사용하여 모델을 다시 실행하고 결과물 출력 
response = bedrock_runtime.invoke_model(body=body, modelId="amazon.titan-tg1-large")
response_body = json.loads(response.get("body").read())
print(response_body["results"][0]["outputText"]) # 최종 결과 출력 

response = bedrock_runtime.invoke_model(body=body, modelId="amazon.titan-tg1-large")
response_body = json.loads(response.get("body").read())
print(response_body["results"][0]["outputText"])