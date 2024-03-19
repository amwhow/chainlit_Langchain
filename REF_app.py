from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory

from chainlit.types import ThreadDict
import chainlit as cl


def setup_runnable():
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    model = ChatOpenAI(streaming=True)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful chatbot"),
            # Oftentimes inputs to prompts can be a list of messages. This is when you would use a MessagesPlaceholder. These objects are parameterized by a variable_name argument. The input with the same value as this variable_name value should be a list of messages.
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    # This is a simple parser that extracts the content field from an AIMessageChunk, giving us the token returned by the model.
    parser = StrOutputParser()
    print("in set_runnable: ", memory)
    print("in set_runnable: ", model)
    print("in set_runnable: ", prompt)
    runnable = (
        RunnablePassthrough.assign(
            # 
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | parser
    )
    cl.user_session.set("runnable", runnable)


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
async def on_chat_start():
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))
    print("chat start - cl.user_session.memory: ", cl.user_session.get("memory"))
    print("chat start - loaded memory var: ", cl.user_session.get("memory").load_memory_variables({}))
    setup_runnable()

# update memory using the history from thread and then call the runnable to process on the updated chain!
@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    memory = ConversationBufferMemory(return_messages=True)
    print("memory in chat_resume: ", memory)
    print("thread in chat_resume: ", thread)
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        print("message in the loop: ", message)
        if message["type"] == "USER_MESSAGE":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    print("chat resume - cl.user_session.memory: ", cl.user_session.get("memory"))
    print("chat resume - loaded memory var: ", cl.user_session.get("memory").load_memory_variables({}))
    setup_runnable()


@cl.on_message
async def on_message(message: cl.Message):
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    res = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)