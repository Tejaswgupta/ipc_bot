import json
import re

import asyncpg
import chainlit as cl
import qdrant_client
import torch
from langchain.agents import (AgentExecutor, AgentType, Tool, ZeroShotAgent,
                              initialize_agent)
from langchain.agents.agent_toolkits import (
    create_conversational_retrieval_agent, create_retriever_tool)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain, LLMChain, RetrievalQA
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFaceTextGenInference, OpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.retrievers import (BM25Retriever,
                                  ContextualCompressionRetriever,
                                  EnsembleRetriever)
from langchain.retrievers.document_compressors import (EmbeddingsFilter,
                                                       LLMChainExtractor,
                                                       LLMChainFilter)
from langchain.schema import Document, StrOutputParser
from langchain.schema.messages import SystemMessage
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain.tools import tool
from langchain.vectorstores import FAISS, Qdrant
# %%
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# @cl.set_chat_profiles
# async def chat_profile():
#     return [
#         cl.ChatProfile(
#             name="GPT-3.5",
#             markdown_description="The underlying LLM model is **GPT-3.5**.",
#             icon="https://picsum.photos/200",
#         ),
#         cl.ChatProfile(
#             name="GPT-4",
#             markdown_description="The underlying LLM model is **GPT-4**.",
#             icon="https://picsum.photos/250",
#         ),
#     ]


@cl.on_chat_start
async def on_chat_start():
    client = qdrant_client.QdrantClient(
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
    )

    conn = await asyncpg.connect(host="legalscraperserver.postgres.database.azure.com",
                                 database="postgres",
                                 user="tejasw",
                                 password="Password1234",
                                 port=5432,)
    row = await conn.fetch(
        '''SELECT sections, act_name, text
    FROM acts
    WHERE sections IS NOT NULL
    AND(act_name LIKE '%The Indian Penal Code, 1860%' OR act_name LIKE '%The Code of Criminal Procedure, 1973%' OR act_name LIKE '%Indian Evidence Act%');
    ''')

    def remove_between_periods(sentence):
        # Define a regular expression pattern to match text between two periods
        pattern = r'\.(.*?)\.'

        # Use re.sub to replace the matched substring with an empty string
        modified_sentence = re.sub(pattern, '.', sentence)
        modified_sentence = re.sub(r'\.', '', modified_sentence)

        return modified_sentence

    def preprocess_text(text):
        text = re.sub(r'(?:\n\s*)+', '\n', text)
        text = re.sub(r'\*', '', text)

        return text

    def process(act):
        data = []
        for section in act['sections']:
            json_data = json.loads(section)
            # if 'omitted.' in json_data['section_name'].lower():
            # continue
            json_data['section_name'] = f"{remove_between_periods(json_data['section_name'])} of {act['act_name'].replace(',','')}"
            d = json_data['section_name'] + ' : ' + json_data['text']
            data.append({
                'name': json_data['section_name'],
                'content': preprocess_text(d),
                # 'section_name': json_data['section_name'],
            })
        return data

    ipc_loader = [Document(page_content=s['content'], metadata={
        'name': s['name']}) for s in process(row[0])]
    iea_loader = [Document(page_content=s['content'], metadata={
        'name': s['name']}) for s in process(row[1])]
    crpc_loader = [Document(page_content=s['content'], metadata={
                            'name': s['name']}) for s in process(row[2])]

    with open('bsa.json', 'r') as bsa:
        bsa_loader = [
            Document(
                page_content=f"{s['section_name']} of Bharatiya Sakshya Adhiniyam(BSA): {s['content']}",
                metadata={
                    'name': f"{s['section_name']} of Bharatiya Sakshya Adhiniyam(BSA)"}) for s in json.load(bsa)]

    with open('bns.json', 'r') as bns:
        bns_loader = [
            Document(
                page_content=f"{s['section_name']} of Bharatiya Nyaya Sanhita(BNS): {s['content']}",
                metadata={
                    'name': f"{s['section_name']} of Bharatiya Nyaya Sanhita(BNS)"}) for s in json.load(bns)]

    with open('bnss.json', 'r') as bnss:
        bnss_loader = [
            Document(
                page_content=f"{s['section_name']} of Bharatiya Nagarik Suraksha Sanhita(BNSS): {s['content']}",
                metadata={
                    'name': f"{s['section_name']} of Bharatiya Nagarik Suraksha Sanhita(BNSS)"}) for s in json.load(bnss)]

    model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True}

    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available()
                      else 'mps'},
        encode_kwargs=encode_kwargs
    )

    ipc_db = Qdrant(
        client=client, collection_name="ipc",
        embeddings=model_norm
    )

    crpc_db = Qdrant(
        client=client, collection_name="crpc",
        embeddings=model_norm
    )

    iea_db = Qdrant(
        client=client, collection_name="iea",
        embeddings=model_norm
    )

    bnss_db = Qdrant(
        client=client, collection_name="bnss",
        embeddings=model_norm
    )

    bsa_db = Qdrant(
        client=client, collection_name="bsa",
        embeddings=model_norm
    )

    bns_db = Qdrant(
        client=client, collection_name="bns",
        embeddings=model_norm
    )

    prefix = """You are Votum, an expert legal assistant with extensive knowledge about Indian law. Your task is to respond with the description of the section if provided with a section number OR respond with section number if given a description.
Remember the following while answering any query:
- The Bharatiya Sakshya Adhiniyam (BSA) will be replacing The Indian Evidence Act (IEA).
- The Bharatiya Nyaya Sanhita (BNS) will be replacing The Indian Penal Code (IPC).
- The Bharatiya Nagarik Suraksha Sanhita Sanhita(BNSS) will be replacing the Code of Criminal Procedure (CrPC).
Whenever asked regarding about a section of an act that has been replaced , first lookup the definition using it's respective tool , followed by searching the returned description with the newer alternative's tool. If asked about a query, only use one of the newer act tools.
-----
Steps overview:
- Query: IPC Section 289
- Use search_ipc with input IPC Section 289
- Invoke search_bns tool with the description received from the last step.
- Analyze and respond appropirately.
-----
You have access to the following tools:
"""
    openai_llm = AzureChatOpenAI(
        # model="gpt-4-turbo",
        deployment_name="gpt-4-turbo",
        api_key="70ecbc470b4942c6971bf2109a0003b2",
        azure_endpoint="https://votum.openai.azure.com/",
        api_version="2023-07-01-preview",
        streaming=True,
    )

    mistral_llm = ChatOpenAI(openai_api_base='http://20.124.240.6:8083/v1',
                             openai_api_key='none',
                             model='TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ',
                             temperature=0.7)

    @tool('search_ipc')
    def search_ipc(query: str) -> str:
        """Query tool to search and return the relevant Indian Penal Code (IPC) section name and description."""
        bm25_retriever = BM25Retriever.from_documents(ipc_loader)
        bm25_retriever.k = 4
        retriever = ipc_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 4})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        res = ensemble_retriever.get_relevant_documents(query)
        return res

    @tool('search_iea')
    def search_iea(query: str) -> str:
        """Query tool to search and return the relevant The Indian Evidence Act (IEA) section name and description."""
        bm25_retriever = BM25Retriever.from_documents(iea_loader)
        bm25_retriever.k = 4
        retriever = iea_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 4})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        res = ensemble_retriever.get_relevant_documents(query)
        return res

    @tool('search_crpc')
    def search_crpc(query: str) -> str:
        """Query tool to search and return the relevant Code of Criminal Procedure (CRPC) section name and description."""
        bm25_retriever = BM25Retriever.from_documents(crpc_loader)
        bm25_retriever.k = 4
        retriever = crpc_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 4})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        res = ensemble_retriever.get_relevant_documents(query)
        return res

    @tool('search_bsa')
    def search_bsa(query: str) -> str:
        """Query tool to search and return the relevant Bharatiya Sakshya Adhiniyam(BSA) section name and description."""
        bm25_retriever = BM25Retriever.from_documents(bsa_loader)
        bm25_retriever.k = 4
        retriever = bsa_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 4})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        res = ensemble_retriever.get_relevant_documents(query)
        return res

    @tool('search_bns')
    def search_bns(query: str) -> str:
        """Query tool to search and return the relevant Bharatiya Nyaya Sanhita (BNS) section name and description."""
        bm25_retriever = BM25Retriever.from_documents(bns_loader)
        bm25_retriever.k = 4
        retriever = bns_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 4})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        res = ensemble_retriever.get_relevant_documents(query)
        return res

    @tool('search_bnss')
    def search_bnss(query: str) -> str:
        """Query tool to search and return the relevant Bharatiya Nagarik Suraksha Sanhita Sanhita(BNSS) section name and description."""
        bm25_retriever = BM25Retriever.from_documents(bnss_loader)
        bm25_retriever.k = 4
        retriever = bnss_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 4})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, retriever], weights=[0.5, 0.5]
        )
        res = ensemble_retriever.get_relevant_documents(query)
        return res

    template = """This is a conversation between a human and a bot:

        {chat_history}

        Write a summary of the conversation for {input}:
        """

    prompt = PromptTemplate(
        input_variables=["input", "chat_history"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    # chat_profile = cl.user_session.get("chat_profile")

    # print(chat_profile)
    summary_chain = LLMChain(
        # llm=mistral_llm if chat_profile == 'GPT-3.5' else openai_llm,
        llm=openai_llm,
        prompt=prompt,
        verbose=True,
        memory=readonlymemory,
    )

    tools = [
        search_ipc,
        search_iea,
        search_crpc,
        search_bsa,
        search_bns,
        search_bnss,
        Tool(
            name="Summary",
            func=summary_chain.run,
            description="useful for when you summarize a conversation. The input to this tool should be a string, representing who will read this summary.",
        ),
    ]

    suffix = """Begin!"

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm_chain = LLMChain(
        # llm=mistral_llm if chat_profile =='GPT-3.5' else openai_llm,
        llm=openai_llm,
        prompt=prompt,
    )
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)
    await cl.make_async(agent.run)(message.content, callbacks=[cb])
