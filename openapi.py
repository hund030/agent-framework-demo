import asyncio
from typing import Optional
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings, AzureAIAgentThread
from azure.identity.aio import DefaultAzureCredential
from azure.ai.agents.models import OpenApiTool, OpenApiAnonymousAuthDetails, ToolDefinition, ToolResources
import json


def openApiTool():
    with open("todo.json", "r") as f:
        todo_json = json.loads(f.read())

    auth = OpenApiAnonymousAuthDetails()
    openapi_tool = OpenApiTool(
        name="todo",
        description="A tool for managing a todo list",
        auth=auth,
        spec=todo_json,
    )
    return openapi_tool


async def create_agent(client, tools: list[ToolDefinition], tool_resources: Optional[ToolResources] = None):
    # 1. Define an agent on the Azure AI agent service
    agent_definition = await client.agents.create_agent(
        model=AzureAIAgentSettings().model_deployment_name,  # type: ignore
        name="agent_framework",
        instructions="You are a helpful assistant that can answer questions and help with tasks.",
        description="Agent Framework",
        temperature=0.5,
        top_p=0.5,
        tools=tools,
        tool_resources=tool_resources,
    )

    # 2. Create a Semantic Kernel agent based on the agent definition
    agent = AzureAIAgent(
        client=client,
        definition=agent_definition,
    )

    return agent


async def interact_with_agent(agent: AzureAIAgent, message: str):
    thread = AzureAIAgentThread(client=agent.client)

    async for response in agent.invoke(message, thread=thread):
        yield response.content
        thread = response.thread


async def main():
    async with (
        DefaultAzureCredential() as credential,
        AzureAIAgent.create_client(credential=credential) as client,
    ):
        tool = openApiTool()
        tool_definitions = tool.definitions
        agent = await create_agent(client, tool_definitions)  # type: ignore
        print(agent.id)

        async for response in interact_with_agent(agent, "Get all todos items"):
            print(response)

        await client.agents.delete_agent(agent.id)


def run():
    """Synchronous wrapper for the async main function"""
    asyncio.run(main())


if __name__ == "__main__":
    run()
