import os
import importlib
from operator import itemgetter
from typing import List
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.schema.output_parser import StrOutputParser
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.docstore.document import Document
import chardet

import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.types import ThreadDict

index_name = "it-support-helper"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()
namespace = "57d4818d-f14b-4fb4-b950-868cae0f353c"
namespaces = set()

LOADER_DICT = {"TextLoader": ['.txt'],
               "PyPDFLoader": ['.pdf'],
               "BSHTMLLoader": ['.html', '.htm'],
               "MHTMLLoader": ['.mhtml'],
               "UnstructuredMarkdownLoader": ['.md'],
               "JSONLoader": [".json"],
               "JSONLinesLoader": [".jsonl"],
               "CSVLoader": [".csv"],
               # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
               #"RapidOCRPDFLoader": [".pdf"],
               "RapidOCRDocLoader": ['.docx', '.doc'],
               "RapidOCRPPTLoader": ['.ppt', '.pptx', ],
               "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf',  '.xml',
                                          '.epub', '.odt','.tsv'],
               "UnstructuredEmailLoader": ['.eml', '.msg'],
               "UnstructuredEPubLoader": ['.epub'],
               "UnstructuredExcelLoader": ['.xlsx', '.xls', '.xlsd'],
               "NotebookLoader": ['.ipynb'],
               "UnstructuredODTLoader": ['.odt'],
               "PythonLoader": ['.py'],
               "UnstructuredRSTLoader": ['.rst'],
               "UnstructuredRTFLoader": ['.rtf'],
               "SRTLoader": ['.srt'],
               "TomlLoader": ['.toml'],
               "UnstructuredTSVLoader": ['.tsv'],
               "UnstructuredWordDocumentLoader": ['.docx', '.doc'],
               "UnstructuredXMLLoader": ['.xml'],
               "UnstructuredPowerPointLoader": ['.ppt', '.pptx'],
               "EverNoteLoader": ['.enex'],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]

# if there's no connected Pinecone index to the current user - returning a loader
def process_file(file: AskFileResponse):
    loader_kwargs = {}
    ext = get_file_extension(file.name)
    print(f"in process_file(), the ext for {file.name} is {ext}")
    document_loader_name = get_LoaderClass(ext)
    print("document_loader_name: ", document_loader_name)   
    try:
        if document_loader_name in ["RapidOCRLoader", "FilteredCSVLoader"]:
            document_loaders_module = importlib.import_module('document_loaders.*')
        else:
            document_loaders_module = importlib.import_module('langchain_community.document_loaders')
    except Exception as e:
        msg = f"error while finding the Loader:{document_loader_name} for the file {file}. Error message: {e}"
        print(msg)
        document_loaders_module = importlib.import_module('langchain_community.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")
    
    if document_loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif document_loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file.path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif document_loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif document_loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file.path, **loader_kwargs)
    return loader    
    
    # DocumentLoader = getattr(document_loaders_module, document_loader_name)
    # print("DocumentLoader: ", DocumentLoader)

    # loader = DocumentLoader(file.path)
    # documents = loader.load()
    # docs = text_splitter.split_documents(documents)
    # for i, doc in enumerate(docs):
    #     doc.metadata["source"] = f"source_{i}"
    # return docs

def get_file_extension(filename: str) -> str:
    # Split the filename by '.' and return the last part preceded by a dot
    # If there is no dot in the filename, it returns an empty string
    extension = '.' + filename.split('.')[-1] if '.' in filename else ''
    return extension

def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass
        
# check the Pinecone index that is linked to the user's account
if namespace:
    welcome_message = f"""We have found the Pinecone index that is linked to your account - "{index_name}", feel free to ask any question to start!
"""
    upload_message = """Feel free to upload files to your existing Pinecone index to provide more context for your questions.
"""
else:
    welcome_message = """Welcome to the IT Support Helper!
    If this is your first time using this tool, simply upload a PDF or text file and then ask a question to start!
    """
# if found a connected Pinecone index to the current user, process the uploaded files and upload them to current Pinecone namespace
async def process_uploaded_file(files):
    for file in files:
        loader_kwargs = {}
        ext = get_file_extension(file.name)
        print(f"in process_file(), the ext for {file.name} is {ext}")
        document_loader_name = get_LoaderClass(ext)
        print("document_loader_name: ", document_loader_name)     

        try:
            if document_loader_name in ["RapidOCRLoader", "FilteredCSVLoader"]:
                document_loaders_module = importlib.import_module('document_loaders.*')
            else:
                document_loaders_module = importlib.import_module('langchain_community.document_loaders')
            DocumentLoader = getattr(document_loaders_module, document_loader_name)
        except Exception as e:
            msg = f"error while finding the Loader:{document_loader_name} for the file {file}. Error message: {e}"
            print(msg)
            document_loaders_module = importlib.import_module('langchain_community.document_loaders')
            DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

        # if document_loader_name == "UnstructuredFileLoader":
        #     loader_kwargs.setdefault("autodetect_encoding", True)
        # elif document_loader_name == "CSVLoader":
        #     if not loader_kwargs.get("encoding"):
        #         # if encoding is not specified，recognise the encoding type of the file to avoid encoding error from langchain loader
        #         with open(file.path, 'rb') as struct_file:
        #             encode_detect = chardet.detect(struct_file.read())
        #         if encode_detect is None:
        #             encode_detect = {"encoding": "utf-8"}
        #         loader_kwargs["encoding"] = encode_detect["encoding"]

        # elif document_loader_name == "JSONLoader":
        #     loader_kwargs.setdefault("jq_schema", ".")
        #     loader_kwargs.setdefault("text_content", False)
        # elif document_loader_name == "JSONLinesLoader":
        #     loader_kwargs.setdefault("jq_schema", ".")
        #     loader_kwargs.setdefault("text_content", False)

        loader = DocumentLoader(file.path, **loader_kwargs)
        print("DocumentLoader: ", DocumentLoader)
        # print("loader kwargs: ", loader_kwargs)
        # print("loader: ", loader)

        # issue with unstructured loaders here
        # loader = UnstructuredHTMLLoader(file.path, mode='elements', strategy='fast')
        print("loader: ", loader)
        documents = loader.load()
        print("documents: ", documents)
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        docsearch = Pinecone.from_documents(
            docs, embeddings, index_name=index_name, namespace=namespace
        )
        print(f"completed processing file: {file}")
    print(f"`{files}` processed. You can now ask questions!")
    return docsearch  
    # return 1

# process uploaded file if there's no connected Pinecone index to the current user
def get_docsearch(file: AskFileResponse):
    docs = process_file(file)

    # Save data in the user session
    cl.user_session.set("docs", docs)

    # Create a unique namespace for the file
    namespace = file.id

    if namespace in namespaces:
        docsearch = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
    else:
        docsearch = Pinecone.from_documents(
            docs, embeddings, index_name=index_name, namespace=namespace
        )
        namespaces.add(namespace)

    return docsearch

# login control
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def start():
    await cl.Avatar(
        name="Chatbot",
        url="https://avatars.githubusercontent.com/u/128686189?s=400&u=a1d1553023f8ea0921fba0debbe92a8c5f840dd9&v=4",
    ).send()
    
    elements = [cl.Text(name="Welcome to the IT Support Helper!", content=welcome_message, display="inline")]
    
    await cl.Message(
        content="",
        elements=elements,
    ).send()

    identifier = cl.user_session.get("user").identifier
    if identifier == "admin":
        docsearch = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
        print("docsearch: ", docsearch)

        # # test with adding AskFileMessage for existing Pinecone index, how to make an AskFileMessage persist?
        # files = await cl.AskFileMessage(
        #         content=upload_message,
        #         accept=["*"],
        #         max_size_mb=20,
        #         timeout=180,
        #         max_files=10,
        #     ).send()
        # msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
        # await msg.send()

        # docsearch = await cl.make_async(process_uploaded_file(files))

        # # Let the user know that the system is ready
        # msg.content = f"`{files}` processed. You can now ask questions!"
        # await msg.update() 
    
    # if not, user should be asked to upload a doc to start.
    else:
        files = None
        while files is None:
            files = await cl.AskFileMessage(
                content=welcome_message,
                accept=["text/plain", "application/pdf", "text/html"],
                max_size_mb=20,
                timeout=180,
                max_files=10,
            ).send()

        file = files[0]
        cl.user_session.set("file_id", file.id)

        msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
        await msg.send()

        # No async implementation in the Pinecone client, fallback to sync
        docsearch = await cl.make_async(get_docsearch)(file)

        # Let the user know that the system is ready
        msg.content = f"`{file.name}` processed. You can now ask questions!"
        await msg.update() 

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    cl.user_session.set("chain", chain)
    cl.user_session.set("memory", memory)

# update memory using the history from thread and then call the runnable to process on the updated chain!
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
    )

    docsearch = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )
    
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    # the Q&A functioning part
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    # use this updated memory and chain to process user new questions
    cl.user_session.set("chain", chain)
    cl.user_session.set("memory", memory)

@cl.on_message
async def main(message: cl.Message):
    memory = cl.user_session.get("memory") # type: ConversationBufferMemory
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    if message.elements:
        print("file uploaded: ", message.elements)
        files = message.elements
        print("before file is uploaded print.")
        await process_uploaded_file(files)

    print("file should be processed print.")
    cb = cl.AsyncLangchainCallbackHandler()
    res_chain = await chain.acall(message.content, callbacks=[cb])
    answer = res_chain["answer"]
    source_documents = res_chain["source_documents"]  # type: List[Document]
    text_elements = []  # type: List[cl.Text]
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"
    res = cl.Message(content=answer, elements=text_elements)
    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)