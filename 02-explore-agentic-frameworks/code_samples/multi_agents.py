from dotenv import load_dotenv
import os

import autogen
from autogen import ConversableAgent
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.azure import AzureAIChatCompletionClient
from azure.core.credentials import AzureKeyCredential

config_list = [
    {
        "base_url": "https://models.inference.ai.azure.com",
        "model": "gpt-4o-mini",
        "api_key": os.getenv("GITHUB_TOKEN"),
    }
]

llm_config = {
    "config_list": config_list,
}

# creating agents, then create a round robin schedule where they can work together, in this case in order

client = AzureAIChatCompletionClient(
    model="gpt-4o-mini",
    endpoint="https://models.inference.ai.azure.com",
    # To authenticate with the model you will need to generate a personal access token (PAT) in your GitHub settings.
    # Create your PAT token by following instructions here: https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens
    credential=AzureKeyCredential(os.getenv("GITHUB_TOKEN")),
    model_info={
        "json_output": True,
        "function_calling": True,
        "vision": True,
        "family": "unknown",
    },
)


async def run_group_chat():
    boss = ConversableAgent(
        "boss",
        llm_config=False,
        code_execution_config=False,
        max_consecutive_auto_reply=50,
        human_input_mode="NEVER",
    )
    coder = ConversableAgent(
        name="coder",
        llm_config=llm_config,
        system_message="""
You are an expert Python developer.  You are writing a FastAPI application and accepting and implementing review suggestions for same.
You are using the pytest framework for testing.
You are using SQLite for the database, and SQLAlchemy for the ORM.

Lets generate high quality code, to maximise the documentation we get from FastAPI.
You must always write code that is easy to read and understand in every response,
you cannot just take suggestions and not apply them.

Develop multiple code blocks to complete the task.
We are using a standard 3-tier system, with web/, service/ and db/ directories.
Let's capture all exceptions in the 3 tiers into a single exception handler.
Let's use our own APIException class for all exceptions.

Don't forget to include the filename in each code block: ```# filename: <filename>```
Never shorten the response, always include all code for all files generated.
Your responses will be archived to disk, so eliding code is not wanted.
""",
    )

    reviewer = ConversableAgent(
        name="reviewer",
        llm_config=llm_config,
        system_message="""
You are a helpful assistant highly skilled in evaluating the quality of a
fast api data layer, service layer and web endpoint layer given
code blocks.

YOU MUST CONSIDER FAST API BEST PRACTICES for each evaluation.
Specifically, you can carefully evaluate the code across the following
dimensions
- bugs (bugs):  are there bugs, logic errors, syntax error or typos? Are there any reasons why the code may fail to compile? How should it be fixed? If ANY bug exists, the bug score MUST be less than 5.
- exception handling: are all exception caught at the data and servic elayers, and converted into `APiException` subclasses, such that we can have one exception handler?
- docstrings: are all public functions documented with doc strings with good descriptions and examples?
- class comments: does each class have a comment describing its purpose?
- class names: are the class names descriptive and appropriate?
- function names: are the function names descriptive and appropriate?
- function arguments: are the function arguments descriptive and appropriate?
- function return values: are the function return values descriptive and appropriate?
- type hints: are all functions annotated with type hints?
- fast api router organization: does each router have a descriptive name? are the routers organized in a logical way?
- fast api router docstrings: are the routers documented with doc strings with good descriptions and examples?
- fast api router function names: are the fast api router function names descriptive and appropriate?

Do NOT suggest code.
Do NOT accept partial code with pieces elided.
Based on the critique above, suggest a concrete list of actions that the coder should take to improve the code.
""",
    )


    groupchat = autogen.GroupChat(
        agents=[boss, coder, reviewer],
        messages=[],
        max_round=10,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    manager.initiate_chat(
        boss,
        True,
        False,
        message="""
Create a FASTAPI application.  You need to capture model objects for the
following entities: Customer, Book, Order, Category.
A Customer can have many Orders.
An Order can have many Books.
Books can belong to one Category.

We need to find all categories, find books within categories.
Entities should be identified by guid, and have a name.
Entities should have a created_at and updated_at timestamp.
Entities should have a deleted flag.
    """,
    )


if __name__ == "__main__":
    load_dotenv()
    import asyncio

    asyncio.run(run_group_chat())
