import os
import time
from dotenv import load_dotenv
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import McpTool, RequiredMcpToolCall, SubmitToolApprovalAction, ToolApproval

# Load environment variables from .env file
load_dotenv()

def run():
    # Get MCP server configuration from environment variables
    mcp_server_url = os.environ.get(
        "MCP_SERVER_URL", "https://gitmcp.io/Azure/azure-rest-api-specs")
    mcp_server_label = os.environ.get("MCP_SERVER_LABEL", "github")

    # Get Azure project configuration from environment variables
    azure_project_endpoint = os.environ.get("AZURE_AI_PROJECT_ENDPOINT")
    if not azure_project_endpoint:
        raise ValueError("AZURE_AI_PROJECT_ENDPOINT is not set")

    project_client = AIProjectClient(
        endpoint=azure_project_endpoint,
        credential=DefaultAzureCredential(),
    )

    # [START create_agent_with_mcp_tool]
    # Initialize agent MCP tool
    mcp_tool = McpTool(
        server_label=mcp_server_label,
        server_url=mcp_server_url,
        allowed_tools=[],  # Optional: specify allowed tools
    )

    # Get agent configuration from environment variables
    agent_model = os.environ.get("AGENT_MODEL", "gpt-4o")
    agent_name = os.environ.get("AGENT_NAME", "my-mcp-agent")
    agent_instructions = os.environ.get(
        "AGENT_INSTRUCTIONS", 
        "You are a helpful agent that can use MCP tools to assist users. Use the available MCP tools to answer questions and perform tasks."
    )
    # allowed_tools = os.environ.get("ALLOWED_TOOLS", "search_azure_rest_api_code")
    # mcp_tool.allow_tool(allowed_tools)
    # print(f"Allowed tools: {mcp_tool.allowed_tools}")

    # Create agent with MCP tool and process agent run
    with project_client:
        agents_client = project_client.agents

        agent = agents_client.create_agent(
            model=agent_model,
            name=agent_name,
            instructions=agent_instructions,
            tools=mcp_tool.definitions,
        )

        print(f"Created agent, ID: {agent.id}")
        print(f"MCP Server: {mcp_tool.server_label} at {mcp_tool.server_url}")

        thread = agents_client.threads.create()
        print(f"Created thread, ID: {thread.id}")

        user_message = os.environ.get(
            "USER_MESSAGE", 
            "Please summarize the Azure REST API specifications Readme for Microsoft container app resource manager"
        )
        mcp_header_key = os.environ.get("MCP_HEADER_KEY", "SuperSecret")
        mcp_header_value = os.environ.get("MCP_HEADER_VALUE", "123456")

        message = agents_client.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_message
        )
        print(f"Created message, ID: {message.id}")

        mcp_tool.update_headers(mcp_header_key, mcp_header_value)
        mcp_tool.set_approval_mode("never")
        run = agents_client.runs.create(
            thread_id=thread.id, agent_id=agent.id, tool_resources=mcp_tool.resources)
        print(f"Created run, ID: {run.id}")

        while run.status in ["queued", "in_progress", "requires_action"]:
            time.sleep(1)
            run = agents_client.runs.get(thread_id=thread.id, run_id=run.id)

            if run.status == "requires_action" and isinstance(run.required_action, SubmitToolApprovalAction):
                tool_calls = run.required_action.submit_tool_approval.tool_calls
                if not tool_calls:
                    print("No tool calls provided - cancelling run")
                    agents_client.runs.cancel(thread_id=thread.id, run_id=run.id)
                    break

                tool_approvals = []
                for tool_call in tool_calls:
                    if isinstance(tool_call, RequiredMcpToolCall):
                        try:
                            print(f"Approving tool call: {tool_call}")
                            tool_approvals.append(
                                ToolApproval(
                                    tool_call_id=tool_call.id,
                                    approve=True,
                                    headers=mcp_tool.headers,
                                )
                            )
                        except Exception as e:
                            print(f"Error approving tool_call {tool_call.id}: {e}")

                print(f"tool_approvals: {tool_approvals}")
                if tool_approvals:
                    agents_client.runs.submit_tool_outputs(
                        thread_id=thread.id, run_id=run.id, tool_approvals=tool_approvals
                    )

            print(f"Current run status: {run.status}")
            # [END handle_tool_approvals]

        print(f"Run completed with status: {run.status}")
        if run.status == "failed":
            print(f"Run failed: {run.last_error}")

        # Display run steps and tool calls
        run_steps = agents_client.run_steps.list(
            thread_id=thread.id, run_id=run.id)

        # Loop through each step
        for step in run_steps:
            print(f"Step {step['id']} status: {step['status']}")

            # Check if there are tool calls in the step details
            step_details = step.get("step_details", {})
            tool_calls = step_details.get("tool_calls", [])

            if tool_calls:
                print("  MCP Tool calls:")
                for call in tool_calls:
                    print(f"    Tool Call ID: {call.get('id')}")
                    print(f"    Type: {call.get('type')}")

            print()  # add an extra newline between steps

        # Fetch and log all messages
        messages = agents_client.messages.list(thread_id=thread.id)
        print("\nConversation:")
        print("-" * 50)
        for msg in messages:
            if msg.text_messages:
                last_text = msg.text_messages[-1]
                print(f"{msg.role.upper()}: {last_text.text.value}")
                print("-" * 50)

        # Clean-up and delete the agent once the run is finished.
        # NOTE: Comment out this line if you plan to reuse the agent later.
        # agents_client.delete_agent(agent.id)
        # print("Deleted agent")
