# Customer Support Agent

An agentic customer-support assistant built with Amazon Bedrock AgentCore and the Strands Agents SDK.

It combines trusted knowledge retrieval, MCP tools, long-term customer memory, code execution, and browser automation to handle customer-support requests.

## Features

- answers product, loyalty, and return-policy questions using an Amazon Bedrock Knowledge Base
- tracks orders and retrieves customer profiles through an AgentCore Gateway
- processes refund requests through gateway tools
- calculates loyalty discounts with AgentCore Code Interpreter
- uses AgentCore Browser for public web lookups
- remembers customer preferences across separate sessions with AgentCore Memory

## Architecture

```text
Customer request
      │
      ▼
AgentCore Runtime ── Strands Agent using Amazon Nova
      ├── AgentCore Gateway ── Order / Customer / Refund tools
      ├── Bedrock Knowledge Base ── product_catalog.txt
      ├── AgentCore Memory ── customer facts and preferences
      ├── Code Interpreter ── loyalty calculations
      └── Browser ── public web information
```

The agent uses tools instead of inventing customer, order, refund, discount, or policy data. If required information is unavailable, it asks one clear follow-up question.

## Project Structure

| Path | Purpose |
| --- | --- |
| `main.py` | AgentCore runtime entry point and Strands agent configuration |
| `lambda/` | Backend functions used by gateway tools |
| `product_catalog.txt` | Knowledge Base source document |
| `run_rubric_tests.py` | Automated end-to-end rubric validation |
| `screenshots/` | Submission evidence |
| `reflection.docx` | Project reflection |
| `pyproject.toml` and `uv.lock` | Python dependencies managed by uv |

## Prerequisites

- AWS credentials configured for `us-east-1`
- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- AWS CLI v2
- Docker Desktop with Buildx for container deployment

## Local Setup

Run these commands from the `starter` directory:

```powershell
uv sync --python 3.13
uv run python -m py_compile main.py
```

The compile command checks that the AgentCore entry point is valid before packaging or deployment.

## AWS Services Used

- Amazon Bedrock AgentCore Runtime
- Amazon Bedrock AgentCore Gateway
- Amazon Bedrock Knowledge Base
- Amazon Bedrock AgentCore Memory
- Amazon Bedrock AgentCore Code Interpreter
- Amazon Bedrock AgentCore Browser
- Amazon ECR
- AWS Lambda
- Amazon API Gateway

## Deployment

AgentCore runtime containers must target Linux ARM64.

Build and push the image to Amazon ECR:

```powershell
docker buildx build `
  --platform linux/arm64 `
  --provenance=false `
  -f .bedrock_agentcore\customer_support_agent\Dockerfile `
  -t <ecr-repository>:<tag> `
  --push .
```

Then create or update the AgentCore runtime using the new image. Wait until the runtime status is `READY` before invoking the default endpoint.

For future code changes:

1. build and push a new image tag
2. update the AgentCore runtime
3. wait for `READY`
4. run the automated tests again

## Rubric Validation

Run the automated validation script:

```powershell
uv run python run_rubric_tests.py
```

The script validates all seven required capabilities:

1. Knowledge Base retrieval for the electronics return policy
2. Order-tracking gateway tool
3. Customer-profile gateway tool
4. Refund-processor gateway tool
5. Code Interpreter loyalty-discount calculation
6. Browser access to `https://example.com`
7. Long-term memory across separate sessions

Results are displayed in a screenshot-friendly format and saved locally under `.test-artifacts/`.

## Evidence

- validation screenshots are in `screenshots/`
- the written reflection is in `reflection.docx`

## Security and Production Considerations

- do not commit AWS credentials, tokens, generated configuration, or temporary request and response files
- apply least-privilege IAM permissions to the AgentCore runtime and gateway targets
- replace mock order and customer data with secured production data sources
- require authentication and authorization before exposing the agent publicly
- add monitoring, tracing, rate limits, error handling, and human escalation paths before production use
