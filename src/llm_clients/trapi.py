import os 
from dotenv import load_dotenv

load_dotenv()
from azure.ai.inference import ChatCompletionsClient
from openai.lib.azure import AzureOpenAI
import pipeline.globals as g
from azure.identity import (
    ChainedTokenCredential,
    AzureCliCredential,
    ManagedIdentityCredential,
    get_bearer_token_provider,
)
class LLMAgent:
    def __init__(
            self,
            api_version,
            model_name,
            model_version,
            deployment_name):
        self.scope = os.getenv("SCOPE", "")
        self.credential = get_bearer_token_provider(
        ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
        self.scope,
    )
        self.model_name = g.TRAPI_DEPLOYMENT_NAME
        self.instance = g.TRAPI_INSTANCE
        self.api_version = api_version
        self.endpoint = f"{g.TRAPI_ENDPOINT_PREFIX}{self.instance}"
        self.llm_client = AzureOpenAI(
        azure_endpoint=self.endpoint,
        azure_ad_token_provider=self.credential,
        api_version=self.api_version,
    )

    def get_llm_response(self, messages):
        response = self.llm_client.chat.completions.create(
            model=self.model_name,
            messages=messages
        )
        return response
    
    @staticmethod
    def trapi_mk_client() -> AzureOpenAI:
        scope = os.getenv("SCOPE", "")
        credential = get_bearer_token_provider(
            ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential()),
            scope,
        )
        return AzureOpenAI(
            azure_endpoint=f"{g.TRAPI_ENDPOINT_PREFIX}{g.TRAPI_INSTANCE}",
            azure_ad_token_provider=credential,
            api_version=g.TRAPI_API_VERSION,
        )

