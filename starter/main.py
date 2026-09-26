"""
Customer Support AI Agent — Starter Code
==========================================
Your task is to complete this file by implementing all sections marked
with # TODO comments.

Reference the project instructions and rubric for guidance.
Work through each section yourself.

Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
# These imports are provided. Do not remove them.
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
from strands_tools.browser import AgentCoreBrowser


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── App Initialisation ────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()


# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"



# ── Configuration ─────────────────────────────────────────────────────────────
GATEWAY_URL = (
    "https://customersupportgateway-pgecbbqjmp."
    "gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
)
KB_ID = "ZPOP6LJE3Y"
REGION = "us-east-1"
MEMORY_ID = "CustomerSupportMemory-f111vu55SB"



# ── Model and Clients ─────────────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client(
    "bedrock-agent-runtime",
    region_name=REGION,
)


# ── Discover the namespaces configured in AgentCore Memory ─────────────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return strategy type -> namespace template."""
    strategies = mem_client.get_memory_strategies(memory_id)
    namespaces = {}

    for strategy in strategies:
        strategy_type = strategy.get("type")
        templates = strategy.get("namespaceTemplates") or strategy.get("namespaces") or []

        if strategy_type and templates:
            namespaces[strategy_type] = templates[0]

    return namespaces



# ── Add persistent customer context to each conversation ─────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    def __init__(self, actor_id, session_id, memory_client, memory_id):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    @staticmethod
    def _get_text(message) -> str:
        """Extract plain text from a Strands message."""
        if not isinstance(message, dict):
            return ""

        content = message.get("content", [])
        if isinstance(content, str):
            return content

        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                text_parts.append(block["text"])

        return "\n".join(text_parts).strip()

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve long-term memory and prepend it to a user message."""
        message = event.message
        content = message.get("content", [])

        if any(
             isinstance(block, dict) and "toolResult" in block
             for block in content
        ):
         return

        # Only enrich plain user messages, never assistant/tool messages.
        if not isinstance(message, dict) or message.get("role") != "user":
            return

        query = self._get_text(message)
        if not query:
            return

        context_items = []

        for strategy_type, template in self.namespaces.items():
            namespace = template.format(
                actorId=self.actor_id,
                sessionId=self.session_id,
            )

            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=query,
                    top_k=5,
                )

                for memory in memories:
                    if memory:
                        context_items.append(f"[{strategy_type}] {memory}")

            except Exception as error:
                logger.warning(
                    "Could not retrieve %s memory: %s",
                    strategy_type,
                    error,
                )

        if not context_items:
            return

        enriched_text = (
            "Customer Context:\n"
            + "\n".join(context_items)
            + f"\n\nCustomer message:\n{query}"
        )

        message["content"] = [{"text": enriched_text}]

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the latest user question and assistant answer as a memory event."""
        messages = event.agent.messages
        customer_query = ""
        agent_response = ""

        for message in reversed(messages):
            text = self._get_text(message)

            if not text:
                continue

            if not agent_response and message.get("role") == "assistant":
                agent_response = text
            elif not customer_query and message.get("role") == "user":
                customer_query = text

            if customer_query and agent_response:
                break

        if not customer_query or not agent_response:
            return

        try:
            self.memory_client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=[
                    (customer_query, "USER"),
                    (agent_response, "ASSISTANT"),
                ],
            )
        except Exception as error:
            logger.warning("Could not save support interaction: %s", error)

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(
            MessageAddedEvent,
            self.retrieve_customer_context,
        )
        registry.add_callback(
            AfterInvocationEvent,
            self.save_support_interaction,
        )


# — Knowledge Base Tool ─────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """Search customer support policies, product details, and loyalty information."""
    try:
        response = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )

        results = response.get("retrievalResults", [])

        if not results:
            return "No relevant information was found in the knowledge base."

        return "\n---\n".join(
            result["content"]["text"]
            for result in results
            if result.get("content", {}).get("text")
        )

    except Exception as error:
        logger.error("Knowledge base search failed: %s", error)
        return "Knowledge base search is temporarily unavailable."

# — Loyalty Discount Tool (Code Interpreter) ────────────────────────
@tool
def calculate_loyalty_discount(order_total: float, tier: str) -> str:
    """Calculate a customer loyalty discount using AgentCore Code Interpreter."""
    discounts = {
        "bronze": 0.00,
        "silver": 0.05,
        "gold": 0.10,
        "platinum": 0.15,
    }

    normalized_tier = tier.strip().lower()
    discount_rate = discounts.get(normalized_tier)

    if discount_rate is None:
        return f"Unknown loyalty tier: {tier}"

    # Fallback calculation if Code Interpreter is unavailable.
    discount_amount = round(order_total * discount_rate, 2)
    final_total = round(order_total - discount_amount, 2)

    code = f"""
order_total = {order_total}
discount_rate = {discount_rate}

discount_amount = round(order_total * discount_rate, 2)
final_total = round(order_total - discount_amount, 2)

print(
    f"Tier: {normalized_tier.title()}\\n"
    f"Discount: ${{discount_amount:.2f}}\\n"
    f"Final total: ${{final_total:.2f}}"
)
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

            for event in response["stream"]:
                if "result" in event:
                    return json.dumps(event["result"])

    except Exception as error:
        logger.warning("Code Interpreter unavailable: %s", error)

    return (
        f"Tier: {normalized_tier.title()}\n"
        f"Discount: ${discount_amount:.2f}\n"
        f"Final total: ${final_total:.2f}"
    )


# — Agent Entrypoint ─────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """Run the customer support agent."""
    prompt = payload.get("prompt", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Provide a non-empty 'prompt'."}

    actor_id = (
    payload.get("customer_id")
    or payload.get("actor_id")
    or "default_customer"
)
    session_id = (
        payload.get("session_id")
        or getattr(context, "session_id", None)
        or str(uuid.uuid4())
    )

    memory_hook = MemoryHook(
        actor_id=actor_id,
        session_id=session_id,
        memory_client=memory_client,
        memory_id=MEMORY_ID,
    )

    browser_tool = AgentCoreBrowser(region=REGION)

    mcp_client = MCPClient(
        lambda: streamable_http_client(GATEWAY_URL)
    )

    system_prompt = """
You are a helpful customer support agent.

Use gateway tools for order tracking, customer profiles, returns, and refunds.
Use search_knowledge_base for product, return-policy, and loyalty questions.
Use calculate_loyalty_discount for discount calculations.
Use the browser only when current public web information is needed.

Never invent order details, refund status, policies, or discounts.
If required information is missing, ask one clear follow-up question.

Browser tool rule:
When calling browser.init_session, the session_name must use only lowercase letters, numbers, and hyphens.
Use a name such as "browser-session". Never use underscores, spaces, or capital letters.
"""

    try:
        # Gateway tools are only available while this connection is open.
        with mcp_client:
            gateway_tools = mcp_client.list_tools_sync()

            agent = Agent(
                model=model,
                tools=[
                    *gateway_tools,
                    search_knowledge_base,
                    calculate_loyalty_discount,
                    browser_tool.browser,
                ],
                hooks=[memory_hook],
                system_prompt=system_prompt,
            )

            result = agent(prompt)

            return {
                "result": result.message["content"][0]["text"],
                "session_id": session_id,
            }

    except Exception as error:
        logger.exception("Agent invocation failed")
        return {"error": f"Agent invocation failed: {str(error)}"}


if __name__ == "__main__":
    app.run()