"""
Configuración multi-entorno: Local → Streamlit Cloud → AWS
"""
# import os
from pathlib import Path
from typing import Optional
import streamlit as st


class BaseConfig:
    """Configuración base para todos los entornos"""
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DOCUMENTS_DIR = BASE_DIR / "documents"
    
    # OpenAI
    OPENAI_MODEL = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE = 0.1
    
    # LangChain
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SIMILARITY_SEARCH_K = 3
    
    # Streamlit
    PAGE_TITLE = "🌱 Syngenta Safety Assistant"
    PAGE_ICON = "🌱"
    
    def __init__(self):
        self.OPENAI_API_KEY = self._get_openai_key()
    
    def _get_openai_key(self) -> Optional[str]:
        """Obtener API key desde diferentes fuentes según el entorno"""
        
        # 1. Variable de entorno (AWS/Docker)
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key
        
        # 2. Streamlit secrets (local/cloud)
        try:
            if hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
                return st.secrets["OPENAI_API_KEY"]
        except Exception:
            pass
        
        # 3. AWS Secrets Manager (futuro)
        if self._is_aws_environment():
            return self._get_aws_secret("syngenta/openai-api-key")
        
        return None
    
    def _is_aws_environment(self) -> bool:
        """Detectar si estamos en AWS"""
        return bool(os.getenv("AWS_EXECUTION_ENV") or os.getenv("ECS_CONTAINER_METADATA_URI"))
    
    def _get_aws_secret(self, secret_name: str) -> Optional[str]:
        """Obtener secret de AWS (para migración futura)"""
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId=secret_name)
            return response['SecretString']
            
        except (ImportError, ClientError):
            return None
    
    def is_configured(self) -> bool:
        """Verificar si la configuración está completa"""
        return bool(self.OPENAI_API_KEY)


class LocalConfig(BaseConfig):
    """Configuración para desarrollo local"""
    ENV = "local"
    DEBUG = True


class StreamlitCloudConfig(BaseConfig):
    """Configuración para Streamlit Cloud"""
    ENV = "streamlit_cloud"
    DEBUG = False


class AWSConfig(BaseConfig):
    """Configuración para AWS (migración futura)"""
    ENV = "aws"
    DEBUG = False
    
    def __init__(self):
        super().__init__()
        # Configuraciones específicas de AWS
        self.AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def get_config() -> BaseConfig:
    """Factory para obtener configuración según entorno"""
    
    # Detectar entorno automáticamente
    if os.getenv("AWS_EXECUTION_ENV"):
        return AWSConfig()
    elif os.getenv("STREAMLIT_SHARING"):
        return StreamlitCloudConfig()
    else:
        return LocalConfig()


# Instancia global
config = get_config()