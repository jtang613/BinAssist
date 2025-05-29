import json
import openai
from .core.settings import get_settings_manager
import requests
import urllib3

class EmbeddingService:
    def __init__(self):
        self.settings = get_settings_manager()
        self.api_provider = self.get_active_provider()

    def get_active_provider(self):
        """
        Returns the currently active API provider.
        """
        active_name = self.settings.get_string('active_provider')
        providers = self.settings.get_json('api_providers', [])
        return next((p for p in providers if p['api___name'] == active_name), None)

    def get_embedding(self, text):
        self.api_provider = self.get_active_provider()
        provider_name = self.api_provider['api___name'].upper()
        if provider_name == 'OPENAI':
            return self._get_openai_embedding(text)
        elif provider_name == 'OLLAMA':
            return self._get_ollama_embedding(text)
        else:
            return self._get_openai_embedding(text)
            #return []

    def _get_openai_embedding(self, text):
        api_key = self.api_provider['api_key']
        api_url = self.api_provider['api__host']
        api_model = self.api_provider['api__model']
        openai.api_key = api_key
        openai.api_base = api_url
        response = openai.embeddings.create(
            input=[text],
            model='text-embedding-ada-002'
        )
        print(f"response.data[0].embedding: {response.data}[0].embedding")
        return response.data[0].embedding

    def _get_ollama_embedding(self, text):
        api_key = self.api_provider['api_key']
        api_url = self.api_provider['api__host'].replace('v1/','/api/')
        api_model = self.api_provider['api__model']
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }
        payload = {
            'model': api_model,
            'keep_alive': 0,
            'prompt': text,
            'options': {}
        }
        # Disable SSL warnings if needed
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.post(
            #f"{api_url}/embeddings",
            "https://192.168.26.201:443/ollama/api/embeddings",
            headers=headers,
            json=payload,
            verify=False  # Set to True in production
        )
        if response.status_code != 200:
            raise Exception(f"Failed to get embedding: {response.status_code} {response.text}")
        response_json = response.json()
        embedding = response_json.get('embedding', [])
        return embedding
