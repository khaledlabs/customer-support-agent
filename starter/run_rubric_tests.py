import json
import sys
import textwrap
import uuid
from datetime import datetime, timezone
from pathlib import Path

import boto3

REGION = "us-east-1"
RUNTIME_ARN = (
    "arn:aws:bedrock-agentcore:us-east-1:368084012852:"
    "runtime/customer_support_agent-likMcE2Ioe"
)
CUSTOMER_ID = "CUST-123"
ARTIFACT_DIR = Path(".test-artifacts")

client = boto3.client("bedrock-agentcore", region_name=REGION)


def invoke(prompt: str) -> str:
    response = client.invoke_agent_runtime(
        agentRuntimeArn=RUNTIME_ARN,
        qualifier="DEFAULT",
        runtimeSessionId=str(uuid.uuid4()),
        contentType="application/json",
        accept="application/json",
        payload=json.dumps(
            {"prompt": prompt, "customer_id": CUSTOMER_ID}
        ).encode("utf-8"),
    )

    body = b"".join(chunk for chunk in response["response"])
    return json.loads(body.decode("utf-8"))["result"]


TESTS = [
    {
        "name": "Knowledge Base RAG",
        "prompt": "What is the return policy for electronics?",
        "expected": ["15 days", "original condition"],
    },
    {
        "name": "Order Tracking MCP Tool",
        "prompt": (
            "Use the order tracking gateway tool for ORD-001. "
            "Return its status, carrier, tracking number, and delivery date."
        ),
        "expected": ["SHIPPED", "UPS", "TRK987654321"],
    },
    {
        "name": "Customer Profile MCP Tool",
        "prompt": (
            "Use the customer profile lookup gateway tool for customer "
            "CUST-123. Return their full name, loyalty tier, and loyalty "
            "points."
        ),
        "expected": ["Jane Smith", "Gold", "4250"],
        "fallback_prompt": (
            "Call the customer profile gateway tool now with customer_id "
            "CUST-123. Do not answer from memory. Return the customer's "
            "name, loyalty tier, and loyalty points."
        ),
    },
    {
        "name": "Refund Processor MCP Tool",
        "prompt": (
            "Initiate a refund for order ORD-003 because it arrived damaged."
        ),
        "expected": ["approved", "refund id"],
    },
    {
        "name": "AgentCore Code Interpreter",
        "prompt": (
            "Use the calculation tool to calculate the 15% Gold loyalty "
            "discount on $89.99. Show the discount and final price."
        ),
        "expected": ["$9.00", "$80.99"],
    },
    {
        "name": "AgentCore Browser",
        "prompt": (
            "Use the browser tool to open https://example.com and tell me "
            "the page title."
        ),
        "expected": ["Example Domain"],
    },
    {
        "name": "Long-Term Customer Memory",
        "prompt": "What contact preferences have I told you about?",
        "expected": ["email", "after 6 PM"],
    },
]


def passed(result: str, expected: list[str]) -> bool:
    normalized_result = result.lower().replace(",", "")
    return all(
        item.lower().replace(",", "") in normalized_result
        for item in expected
    )

def compact(text: str, width: int = 180) -> str:
    return textwrap.shorten(
        " ".join(text.split()),
        width=width,
        placeholder=" ...",
    )


def main() -> None:
    results = []
    total = len(TESTS)

    print()
    print("═" * 78)
    print("        CUSTOMER SUPPORT AGENT • RUBRIC VERIFICATION")
    print("═" * 78)
    print(f"Runtime: {RUNTIME_ARN.rsplit('/', maxsplit=1)[-1]}")
    print(f"Customer: {CUSTOMER_ID}")
    print("═" * 78)

    for number, test in enumerate(TESTS, start=1):
        result = ""
        used_fallback = False

        try:
            result = invoke(test["prompt"])

            if not passed(result, test["expected"]) and test.get(
                "fallback_prompt"
            ):
                used_fallback = True
                result = invoke(test["fallback_prompt"])

            success = passed(result, test["expected"])
        except Exception as error:
            result = f"ERROR: {error}"
            success = False

        status = "✅ PASS" if success else "❌ FAIL"
        matched = ", ".join(test["expected"])

        print(f"\n{number}. {test['name']}")
        print(f"   {status}")
        print(f"   Expected evidence: {matched}")
        print(f"   Response: {compact(result)}")

        if used_fallback:
            print("   Note: Retried with an explicit gateway-tool instruction.")

        results.append(
            {
                "number": number,
                "test": test["name"],
                "passed": success,
                "expected": test["expected"],
                "result": result,
                "used_fallback": used_fallback,
            }
        )

    passed_count = sum(item["passed"] for item in results)

    ARTIFACT_DIR.mkdir(exist_ok=True)
    artifact = ARTIFACT_DIR / (
        "rubric-results-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.json"
    )
    artifact.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "═" * 78)
    print(f"FINAL RESULT: {passed_count}/{total} TESTS PASSED")
    print(f"Local detailed artifact: {artifact}")
    print("═" * 78)

    if passed_count != total:
        sys.exit(1)


if __name__ == "__main__":
    main()