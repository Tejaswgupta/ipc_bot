import json
import re

import asyncpg
import chainlit as cl
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


@cl.on_chat_start
async def on_chat_start():

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

        return modified_sentence

    def preprocess_text(text):
        text = re.sub(r'(?:\n\s*)+', '\n', text)
        text = re.sub(r'\*', '', text)

        return text

    # for act in row:
    #     for section in act['sections']:
    #         json_data = json.loads(section)
    #         # if 'omitted.' in json_data['section_name'].lower():
    #             # continue
    #         json_data['section_name'] = f"{remove_between_periods(json_data['section_name'])} of {act['act_name'].replace(',','')}"
    #         d = json_data['section_name'] + ' : ' + json_data['text']
    #         data.append({
    #             'name': act['act_name'],
    #             'content':preprocess_text(d)
    #         })

    def process(act):
        data = []
        for section in act['sections']:
            json_data = json.loads(section)
            # if 'omitted.' in json_data['section_name'].lower():
            # continue
            json_data['section_name'] = f"{remove_between_periods(json_data['section_name'])} of {act['act_name'].replace(',','')}"
            d = json_data['section_name'] + ' : ' + json_data['text']
            data.append({
                'name':  json_data['section_name'],
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
        bsa_loader = [Document(page_content=f"{s['section_name']} of Bharatiya Sakshya Adhiniyam(BSA): {s['content']}", metadata={
            'name': f"{s['section_name']} of Bharatiya Sakshya Adhiniyam(BSA)"}) for s in json.load(bsa)]

    with open('bns.json', 'r') as bns:
        bns_loader = [Document(page_content=f"{s['section_name']} of Bharatiya Nyaya Sanhita(BNS): {s['content']}", metadata={
            'name': f"{s['section_name']} of Bharatiya Nyaya Sanhita(BNS)"}) for s in json.load(bns)]

    with open('bnss.json', 'r') as bnss:
        bnss_loader = [Document(page_content=f"{s['section_name']} of Bharatiya Nagarik Suraksha Sanhita(BNSS): {s['content']}", metadata={
            'name': f"{s['section_name']} of Bharatiya Nagarik Suraksha Sanhita(BNSS)"}) for s in json.load(bnss)]

    model_name = "BAAI/bge-base-en-v1.5"
    encode_kwargs = {'normalize_embeddings': True}

    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )

    # Here is the nmew embeddings being used
    embedding = model_norm

    ipc_db = Qdrant.from_documents(
        ipc_loader,
        model_norm,
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
        prefer_grpc=True,
        collection_name="ipc",
    )

    crpc_db = Qdrant.from_documents(
        crpc_loader,
        model_norm,
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
        prefer_grpc=True,
        collection_name="crpc",
    )

    iea_db = Qdrant.from_documents(
        iea_loader,
        model_norm,
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
        prefer_grpc=True,
        collection_name="iea",
    )

    bns_db = Qdrant.from_documents(
        bns_loader,
        model_norm,
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
        prefer_grpc=True,
        collection_name="bns",
    )

    bsa_db = Qdrant.from_documents(
        bsa_loader,
        model_norm,
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
        prefer_grpc=True,
        collection_name="bsa",
    )

    bnss_db = Qdrant.from_documents(
        bnss_loader,
        model_norm,
        url="https://70dd04d7-f233-4954-8e4d-54c848c8d13b.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key="xVLVqK38Xf5HYBM4GaKrGy0Csorj1Gc9YJdqn6DARiXD6ES5SuMrfA",
        prefer_grpc=True,
        collection_name="bnss",
    )

    prefix = """You are Votum, an expert legal assistant with extensive knowledge about Indian law. Your task is to respond with the description of the section if provided with a section number OR respond with section number if given a description. 
Remember the following while answering any query:
- The Bharatiya Sakshya Adhiniyam (BSA) will be replacing The Indian Evidence Act (IEA).
- The Bharatiya Nyaya Sanhita (BNS) will be replacing The Indian Penal Code (IPC).
- The Bharatiya Nagarik Suraksha Sanhita Sanhita(BNSS) will be replacing the Code of Criminal Procedure (CrPC).
Whenever asked regarding about a section of an act that has been replaced , first lookup the defination using it's respective tool , followed by searching the returned description with the newer alternative's tool. 
Steps overview:
- Query: IPC Section 289
- Use search_ipc with input IPC Section 289
- Invoke search_bns tool with the description received from the last step.
- Analyze and respond appropirately.
You have access to the following tools:
"""
    llm = AzureChatOpenAI(
        # model="gpt-4-turbo",
        deployment_name="gpt-4-turbo",
        api_key="4e19bb52d70748ec89a517a52303243c",
        azure_endpoint="https://votum.openai.azure.com/",
        api_version="2023-07-01-preview",
    )

    @tool('search_ipc')
    def search_ipc(query: str) -> str:
        """Query tool to search and return the relevant Indian Penal Code (IPC) section name and description."""
        retriever = ipc_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 2})
        res = retriever.get_relevant_documents(query)
        return res

    @tool('search_iea')
    def search_iea(query: str) -> str:
        """Query tool to search and return the relevant The Indian Evidence Act (IEA) section name and description."""
        retriever = iea_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 2})
        res = retriever.get_relevant_documents(query)
        return res

    @tool('search_crpc')
    def search_crpc(query: str) -> str:
        """Query tool to search and return the relevant Code of Criminal Procedure (CRPC) section name and description."""
        retriever = crpc_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 2})
        res = retriever.get_relevant_documents(query)
        return res

    @tool('search_bsa')
    def search_bsa(query: str) -> str:
        """Query tool to search and return the relevant Bharatiya Sakshya Adhiniyam(BSA) section name and description."""
        retriever = bsa_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 2})
        res = retriever.get_relevant_documents(query)
        return res

    @tool('search_bns')
    def search_bns(query: str) -> str:
        """Query tool to search and return the relevant Bharatiya Nyaya Sanhita (BNS) section name and description."""
        retriever = bns_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 2})
        res = retriever.get_relevant_documents(query)
        return res

    @tool('search_bnss')
    def search_bnss(query: str) -> str:
        """Query tool to search and return the relevant Bharatiya Nagarik Suraksha Sanhita Sanhita(BNSS) section name and description."""
        retriever = bnss_db.as_retriever(
            search_type='similarity', search_kwargs={"k": 2})
        res = retriever.get_relevant_documents(query)
        return res

    template = """This is a conversation between a human and a bot:

        {chat_history}

        Write a summary of the conversation for {input}:
        """

    prompt = PromptTemplate(
        input_variables=["input", "chat_history"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")
    readonlymemory = ReadOnlySharedMemory(memory=memory)
    summary_chain = LLMChain(
        llm=llm,
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

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )
    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message: cl.Message):

    agent = cl.user_session.get("agent")  # type: AgentExecutor

    cb = cl.AsyncLangchainCallbackHandler(stream_final_answer=True)
    res = await cl.make_async(agent.run)(message.content, callbacks=[cb])
