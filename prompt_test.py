import boto3
import json
brt = boto3.client(service_name='bedrock-runtime') # bedrock-runtime client를 초기화

# 설정값은 bedrock에서 봤던 옵션 값과 같음
# max_tokens_to_sample : 설정할 최대 토큰 수
# temperature : 낮은 온도는 더 예측이 가능하고, 더 일관된 결과 산출 가능
# top_p : 주어진 상황에서 가능한한 토큰의 분포를 좁히는 데 사용하는 옵션 값
body = json.dumps({
    "prompt": "\n\nHuman: explain black holes to 8th graders\n\nAssistant:",
    "max_tokens_to_sample": 300,
    "temperature": 0.1,
    "top_p": 0.9,
})

modelId = 'anthropic.claude-v2' # 내가 사용할 모델의 id를 지정
# 밑에 둘은 본문의 type을 지정
accept = 'application/json'
contentType = 'application/json'

response = brt.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType) # 모델의 호출에 응답을 받는 코드

response_body = json.loads(response.get('body').read()) # body에서 json 데이터 로드중

# text
print(response_body.get('completion')) # 응답에서 생성된 text를 출력