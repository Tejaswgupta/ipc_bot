import os

from llama_index import (ServiceContext, StorageContext,
                         load_index_from_storage, set_global_service_context)
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import AzureOpenAI

llm = AzureOpenAI(
    model="gpt-35-turbo",
    deployment_name="gpt-3",
    api_key=os.environ["OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ['OPENAI_API_VERSION'],
)
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=os.environ['EMBEDDING_DEPLOYMENT_NAME'],
    api_key=os.environ['OPENAI_API_KEY'],
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_version=os.environ['OPENAI_API_VERSION'],
)


def __get_index(persist_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    index = load_index_from_storage(storage_context)
    return index


def get_query_engines():
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
    )

    set_global_service_context(service_context)

    ipc_act_index = __get_index("./vector_store/ipc")
    nyay_index = __get_index("./vector_store/bns")
    iea_index = __get_index("./vector_store/iea")
    crpc_index = __get_index("./vector_store/crpc")
    bnss_index = __get_index("./vector_store/bnss")
    bs_index = __get_index("./vector_store/bs")

    ipc_act_engine = ipc_act_index.as_chat_engine(verbose=True)
    nyay_engine = nyay_index.as_chat_engine(verbose=True)
    iea_engine = iea_index.as_chat_engine(verbose=True)
    crpc_engine = crpc_index.as_chat_engine(verbose=True)
    bnss_engine = bnss_index.as_chat_engine(verbose=True)
    bs_engine = bs_index.as_chat_engine(verbose=True)

    return ipc_act_engine, nyay_engine, iea_engine, crpc_engine, bnss_engine, bs_engine
