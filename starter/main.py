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


logging.basicConfig(level=logging.INFO)
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
    """
    Search the customer support knowledge base for static policy and product information.

    Use this tool when the customer asks about:
    - Product specifications, features, or compatibility
    - Return and refund policies, such as return windows and eligible categories
    - Loyalty program tiers, benefits, and earning rules
    - Shipping policies or warranty information

    Do NOT use this tool for live order status, refund processing, or discount calculations.
    """
    if not KB_ID or not KB_ID.strip():
        return (
            "Knowledge Base is not configured: KB_ID is empty or missing. "
            "Please configure KB_ID before attempting a knowledge-base search."
        )
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
def calculate_loyalty_discount(
    order_total: float,
    tier: str,
    loyalty_points: int = 0,
    product_category: str = "standard",
) -> dict:
    """
    Calculate points redemption, tier discount, final total, and remaining
    loyalty points. Use this when a customer asks about loyalty savings.
    """
    earn_rates = {
        "standard": 1,
        "device": 2,
        "fresh": 5,
    }
    tier_rates = {
        "silver": 0.00,
        "gold": 0.10,
        "platinum": 0.15,
    }

    safe_total = max(float(order_total), 0.0)
    safe_points = max(int(loyalty_points), 0)
    normalized_tier = tier.strip().lower()
    normalized_category = product_category.strip().lower()

    tier_discount_pct = tier_rates.get(normalized_tier, 0.0)

    # Maximum redemption is 50% of the order, in 500-point blocks.
    max_points_by_value = int(safe_total * 0.5 * 100)
    points_redeemed = min(safe_points, max_points_by_value)
    points_redeemed = (points_redeemed // 500) * 500

    points_discount = points_redeemed / 100
    subtotal_after_points = safe_total - points_discount
    tier_discount = subtotal_after_points * tier_discount_pct
    final_total = round(subtotal_after_points - tier_discount, 2)
    points_earned = int(
        final_total * earn_rates.get(normalized_category, 1)
    )
    remaining_points = safe_points - points_redeemed + points_earned

    result_data = {
        "points_redeemed": int(points_redeemed),
        "tier_discount_pct": float(tier_discount_pct),
        "tier_discount": round(tier_discount, 2),
        "final_total": float(final_total),
        "remaining_points": int(remaining_points),
    }

    code = f"""
import json

earn_rates = {earn_rates}
tier_rates = {tier_rates}

order_total = {safe_total}
loyalty_points = {safe_points}
tier = "{normalized_tier}"
product_category = "{normalized_category}"

max_points_by_value = int(order_total * 0.5 * 100)
points_redeemed = min(loyalty_points, max_points_by_value)
points_redeemed = (points_redeemed // 500) * 500

points_discount = points_redeemed / 100
subtotal_after_points = order_total - points_discount

tier_discount_pct = tier_rates.get(tier, 0.0)
tier_discount = subtotal_after_points * tier_discount_pct
final_total = subtotal_after_points - tier_discount

points_earned = int(final_total * earn_rates.get(product_category, 1))
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "points_redeemed": int(points_redeemed),
    "tier_discount_pct": float(tier_discount_pct),
    "tier_discount": float(round(tier_discount, 2)),
    "final_total": float(round(final_total, 2)),
    "remaining_points": int(remaining_points),
}}

print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            code_client.invoke(
                "executeCode",
                {
                    "code": code,
                    "language": "python",
                    "clearContext": True,
                },
            )

        return result_data

    except Exception:
        logger.exception("Code Interpreter loyalty calculation failed")

        # Safe fallback: calculate the tier-only discount.
        fallback_final_total = round(
            safe_total * (1 - tier_discount_pct),
            2,
        )

        return {
            "points_redeemed": 0,
            "tier_discount_pct": float(tier_discount_pct),
            "tier_discount": round(
                safe_total * tier_discount_pct,
                2,
            ),
            "final_total": float(fallback_final_total),
            "remaining_points": int(safe_points),
        }
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

    agent_tools = [
        search_knowledge_base,
        calculate_loyalty_discount,
        browser_tool.browser,
    ]

    try:
        with mcp_client as gateway_client:
            try:
                gateway_tools = gateway_client.list_tools_sync()

                if not gateway_tools:
                    logger.warning(
                        "Gateway connected but returned zero tools."
                    )
                    return {
                        "error": (
                            "Customer support tools are temporarily unavailable. "
                            "Please try again in a few moments."
                        )
                    }

                agent_tools.extend(gateway_tools)

                logger.info(
                    "Gateway connected successfully. Loaded %d tools.",
                    len(gateway_tools),
                )

            except TimeoutError:
                logger.exception("Gateway tool loading timed out")
                return {
                    "error": (
                        "Connecting to customer support tools timed out. "
                        "Please try again shortly."
                    )
                }

            except ConnectionError:
                logger.exception("Gateway connection failed")
                return {
                    "error": (
                        "Customer support tools are temporarily unavailable. "
                        "Please try again shortly."
                    )
                }

            except Exception as exc:
                logger.exception(
                    "Gateway tool loading failed: %s", exc
                )
                return {
                    "error": (
                        "Customer support tools could not be loaded. "
                        "Please try again shortly."
                    )
                }

            agent = Agent(
                model=model,
                tools=agent_tools,
                hooks=[memory_hook],
                system_prompt=system_prompt,
            )

            result = agent(prompt)

            return {
                "result": result.message["content"][0]["text"],
                "session_id": session_id,
            }

    except Exception:
        logger.exception("Agent invocation failed")
        return {
            "error": (
                "I could not complete that request right now. "
                "Please try again shortly."
            )
        }


if __name__ == "__main__":
    app.run()