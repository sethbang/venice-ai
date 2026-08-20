#!/usr/bin/env python3
"""
Venice AI SDK - Structured Output Example

This example demonstrates how to get structured JSON responses from Venice AI
models using ``chat.completions.parse(response_format=PydanticModel)``. The
``parse()`` method derives the JSON schema from the model class, validates
the response against it, and returns a typed instance — no manual
``json.loads`` + field-presence checks required.

Requirements:
    - Venice AI API key (set as VENICE_API_KEY environment variable)
    - Python 3.13+
    - venice-py SDK (with pydantic available)
"""

import asyncio
import sys
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from venice_ai import SystemMessage, UserMessage, VeniceClient
from venice_ai.exceptions import APIError, APIResponseValidationError, VeniceError

# -----------------------------------------------------------------------------
# Example 1: Math Problem Solver
# -----------------------------------------------------------------------------


class MathStep(BaseModel):
    explanation: str
    output: str


class MathResponse(BaseModel):
    steps: list[MathStep]
    final_answer: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


async def math_problem_example(client: VeniceClient, model: str) -> bool:
    """Demonstrate structured output for solving math problems step-by-step."""
    print("\n📊 Example 1: Math Problem Solver")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    try:
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "You are a helpful math tutor. Always include your confidence "
                        "level (0-1) in your response."
                    )
                ),
                UserMessage(content="Solve the equation: 3x + 15 = 42"),
            ],
            response_format=MathResponse,
            temperature=0.2,
        )
    except ValidationError as e:
        print(f"❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except VeniceError as e:
        # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
        # APITimeoutError, and APIConnectionError. A timeout on a single slow
        # request is tallied as one failed example, not a whole-run crash.
        print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    result = parsed.parsed
    print("✅ Successfully solved math problem!")
    print("\n🔧 Solution Steps:")
    for i, step in enumerate(result.steps, 1):
        print(f"  Step {i}: {step.explanation}")
        print(f"          → {step.output}")

    print(f"\n🏷️ Final Answer: {result.final_answer}")
    if result.confidence is not None:
        print(f"📊 Confidence Level: {result.confidence:.2%}")

    return True


# -----------------------------------------------------------------------------
# Example 2: Data Extraction from Text
# -----------------------------------------------------------------------------


class Headquarters(BaseModel):
    city: str
    country: str


class CompanyInfo(BaseModel):
    company_name: str
    founded_year: int | None = None
    founders: list[str] = Field(default_factory=list)
    industry: str
    headquarters: Headquarters
    key_products: list[str] = Field(default_factory=list)
    employee_count: int | None = None


async def data_extraction_example(client: VeniceClient, model: str) -> bool:
    """Extract structured information from unstructured text."""
    print("\n📊 Example 2: Data Extraction")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    text = """
    Venice AI was founded in 2023 by a team of AI researchers and engineers.
    The company, headquartered in San Francisco, USA, focuses on providing
    accessible AI infrastructure and tools. They offer open-source inference
    servers and a developer-friendly API. The company operates in the artificial
    intelligence industry and has quickly grown to serve thousands of developers worldwide.
    """

    try:
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "Extract structured information from the provided text. "
                        "Only include information explicitly mentioned."
                    )
                ),
                UserMessage(content=f"Extract company information from this text:\n\n{text}"),
            ],
            response_format=CompanyInfo,
        )
    except ValidationError as e:
        print(f"❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except VeniceError as e:
        # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
        # APITimeoutError, and APIConnectionError. A timeout on a single slow
        # request is tallied as one failed example, not a whole-run crash.
        print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    result = parsed.parsed
    print("✅ Successfully extracted company information!")
    print(f"\n🏢 Company: {result.company_name}")
    print(f"🏭 Industry: {result.industry}")
    print(f"📍 Headquarters: {result.headquarters.city}, {result.headquarters.country}")

    if result.founded_year is not None:
        print(f"📅 Founded: {result.founded_year}")
    if result.founders:
        print(f"👥 Founders: {', '.join(result.founders)}")
    if result.key_products:
        print("📦 Key Products:")
        for product in result.key_products:
            print(f"   - {product}")

    return True


# -----------------------------------------------------------------------------
# Example 3: Multiple Choice Quiz Generation
# -----------------------------------------------------------------------------


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: int = Field(ge=0, le=3)
    explanation: str


class Quiz(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"]
    questions: list[QuizQuestion] = Field(min_length=3, max_length=3)


async def quiz_generation_example(client: VeniceClient, model: str) -> bool:
    """Generate a structured quiz with questions and answers."""
    print("\n📊 Example 3: Quiz Generation")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    try:
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "You are a quiz generator. Create educational quizzes with "
                        "clear questions and explanations."
                    )
                ),
                UserMessage(
                    content=(
                        "Create a quiz about Python programming basics with 3 "
                        "questions of medium difficulty."
                    )
                ),
            ],
            response_format=Quiz,
            temperature=0.7,
        )
    except ValidationError as e:
        print(f"❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except VeniceError as e:
        # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
        # APITimeoutError, and APIConnectionError. A timeout on a single slow
        # request is tallied as one failed example, not a whole-run crash.
        print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    quiz = parsed.parsed
    print("✅ Successfully generated quiz!")
    print(f"\n📚 Topic: {quiz.topic}")
    print(f"⚡ Difficulty: {quiz.difficulty}")
    print(f"❓ Questions: {len(quiz.questions)}")
    print("\n" + "-" * 40)

    for i, q in enumerate(quiz.questions, 1):
        print(f"\nQuestion {i}: {q.question}")
        for j, option in enumerate(q.options):
            marker = "✓" if j == q.correct_answer else "  "
            print(f"  {marker} {chr(65 + j)}. {option}")
        print(f"\n💡 Explanation: {q.explanation}")

    return True


# -----------------------------------------------------------------------------
# Example 4: Task Planning and Organization
# -----------------------------------------------------------------------------


class ProjectTask(BaseModel):
    id: str
    name: str
    estimated_hours: float
    dependencies: list[str] = Field(default_factory=list)
    priority: Literal["low", "medium", "high", "critical"]


class ProjectPhase(BaseModel):
    name: str
    description: str
    tasks: list[ProjectTask]


class ProjectRisk(BaseModel):
    description: str
    mitigation: str


class ProjectPlan(BaseModel):
    project_name: str
    total_estimated_hours: float
    phases: list[ProjectPhase]
    risks: list[ProjectRisk] = Field(default_factory=list)


async def task_planning_example(client: VeniceClient, model: str) -> bool:
    """Generate a structured task plan with dependencies and time estimates."""
    print("\n📊 Example 4: Task Planning")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    # The ProjectPlan schema is the deepest in this file (plan → phases[] →
    # tasks[] → risks[]). Soft-structured-output models occasionally emit JSON
    # that drifts from a nested schema — e.g. a risk shaped {"risk": "..."}
    # instead of {"description": ..., "mitigation": ...}. Two reinforcements
    # keep it reliable:
    #   1. Spell the exact field names/types into the prompt (response_format
    #      already sends the schema, but naming the fields in prose measurably
    #      improves compliance) and cap the plan size so the response stays
    #      small and fast — a smaller response is both quicker to generate
    #      (avoiding request timeouts on a big model) and less prone to drift.
    #   2. A bounded retry on ValidationError (below): re-sampling at a low
    #      temperature usually fixes the occasional non-conforming response.
    system_prompt = (
        "You are a project manager. Create concise, realistic project plans. "
        "Return JSON that matches this exact structure:\n"
        "- project_name: string\n"
        "- total_estimated_hours: number\n"
        "- phases: array of objects, each with:\n"
        '    - "name": string\n'
        '    - "description": string\n'
        '    - "tasks": array of objects, each with:\n'
        '        - "id": string (e.g. "T1")\n'
        '        - "name": string\n'
        '        - "estimated_hours": number\n'
        '        - "dependencies": array of task-id strings (use [] if none)\n'
        '        - "priority": one of "low", "medium", "high", "critical"\n'
        "- risks: array of objects, each with EXACTLY two string fields:\n"
        '    - "description": string (what could go wrong)\n'
        '    - "mitigation": string (how to handle it)\n'
        'Do NOT use any other field names for a risk (no bare "risk" field).'
    )
    user_prompt = (
        "Create a project plan for building a simple REST API with user "
        "authentication. Keep it small: exactly 2 phases, 2-3 tasks per phase, "
        "and exactly 2 risks. Each risk must be an object with a 'description' "
        "and a 'mitigation' field."
    )

    # Bounded retry: structured-output models occasionally return JSON that
    # violates the schema. A short retry (re-sampling) usually resolves it.
    # We only retry ValidationError; VeniceError (HTTP/transport) is returned
    # immediately so a genuine API problem isn't masked by pointless retries.
    max_attempts = 3
    parsed = None
    for attempt in range(1, max_attempts + 1):
        try:
            parsed = await client.chat.completions.parse(
                model=model,
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=user_prompt),
                ],
                response_format=ProjectPlan,
                temperature=0.2,
            )
            break
        except ValidationError as e:
            print(
                f"⚠️ Attempt {attempt}/{max_attempts}: model output didn't satisfy "
                f"schema ({type(e).__name__})."
            )
            if attempt == max_attempts:
                print(f"❌ Pydantic ValidationError after {max_attempts} attempts:\n{e}")
                return False
            print("   Retrying...")
        except VeniceError as e:
            # VeniceError is the SDK base: covers APIError, APIResponseValidationError,
            # APITimeoutError, and APIConnectionError. A timeout on a single slow
            # request is tallied as one failed example, not a whole-run crash.
            print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
            return False

    assert parsed is not None  # loop exits only via break (success) or return
    plan = parsed.parsed
    print("✅ Successfully generated project plan!")
    print(f"\n🎯 Project: {plan.project_name}")
    print(f"⏱️ Total Estimated Hours: {plan.total_estimated_hours}")
    print("\n" + "=" * 40)

    for phase in plan.phases:
        print(f"\n📌 Phase: {phase.name}")
        print(f"   {phase.description}")
        print("   Tasks:")
        for task in phase.tasks:
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            print(f"   • [{task.priority.upper()}] {task.name} - {task.estimated_hours}h{deps}")

    if plan.risks:
        print("\n⚠️ Identified Risks:")
        for risk in plan.risks:
            print(f"   • Risk: {risk.description}")
            print(f"     Mitigation: {risk.mitigation}")

    return True


# -----------------------------------------------------------------------------
# Example 5: Error Handling and Validation
# -----------------------------------------------------------------------------


class StrictResponse(BaseModel):
    status: Literal["success", "failure"]
    code: int = Field(ge=100, le=999)
    timestamp: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


async def error_handling_example(client: VeniceClient, model: str) -> bool:
    """Demonstrate error handling for structured outputs.

    ``parse()`` raises ``pydantic.ValidationError`` if the model returns JSON
    that doesn't match the schema (e.g. wrong types, missing required fields,
    pattern mismatch). Other failures bubble up as ``VeniceError`` subclasses
    (``APIError``, ``APIResponseValidationError``, ``APITimeoutError``,
    ``APIConnectionError``) from the HTTP layer.
    """
    print("\n🛡️ Example 5: Error Handling Demo")
    print("=" * 50)
    print(f"🏷️ Using model: {model}")

    try:
        print("📤 Attempting strict structured output request...")
        parsed = await client.chat.completions.parse(
            model=model,
            messages=[
                SystemMessage(
                    content=(
                        "Generate a response with status='success', code=200, and "
                        "current timestamp in ISO format (YYYY-MM-DDTHH:MM:SSZ)"
                    )
                ),
                UserMessage(content="Generate a success response"),
            ],
            response_format=StrictResponse,
            temperature=0.1,
        )
    except ValidationError as e:
        print(f"❌ Pydantic ValidationError: model output didn't satisfy schema:\n{e}")
        return False
    except APIError as e:
        print(f"❌ API Error: {e}")
        print("💡 Tip: Check if the model supports structured output")
        return False
    except APIResponseValidationError as e:
        print(f"❌ Validation Error: {e}")
        print("💡 Tip: Ensure your schema is correctly formatted")
        return False
    except VeniceError as e:
        # Catch-all for the SDK base: APITimeoutError / APIConnectionError are
        # siblings of APIError (not subclasses), so they would otherwise escape
        # the branches above and crash the whole run.
        print(f"❌ Venice API call failed ({type(e).__name__}): {e}")
        return False

    result = parsed.parsed
    print("✅ Valid response received:")
    print(f"   Status: {result.status}")
    print(f"   Code: {result.code}")
    print(f"   Timestamp: {result.timestamp}")
    return True


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------


async def main() -> int:
    """Run all structured output examples. Returns 0 if all pass, 1 if any fail."""
    print("🚀 Venice AI Structured Output Examples")
    print("=" * 60)
    print("Demonstrates chat.completions.parse(response_format=PydanticModel)")
    print("for typed, schema-validated JSON responses.")

    async with VeniceClient() as client:
        print("\n✅ Client initialized successfully")

        # Get a model that supports structured output. The capability filter
        # (require_response_schema) is necessary but not sufficient: members of
        # the venice-uncensored family advertise schema support yet return
        # non-conforming JSON in practice. Exclude the whole family by prefix —
        # computed from the live catalog so we never hardcode specific IDs.
        print("\n🔍 Searching for models with structured output support...")
        all_chat = await client.models.list(type="chat")
        uncensored_family = [m.id for m in all_chat.data if m.id.startswith("venice-uncensored")]
        chat_model = await client.models.resolve_chat(
            require_response_schema=True,
            exclude_models=uncensored_family,
        )
        print(f"📍 Selected model: {chat_model}")

        examples = [
            ("Math Problem Solver", math_problem_example),
            ("Data Extraction", data_extraction_example),
            ("Quiz Generation", quiz_generation_example),
            ("Task Planning", task_planning_example),
            ("Error Handling Demo", error_handling_example),
        ]

        results: list[tuple[str, bool]] = []
        for name, fn in examples:
            ok = await fn(client, chat_model)
            results.append((name, ok))

        passed = sum(1 for _, ok in results if ok)
        failed = len(results) - passed

        print("\n" + "=" * 60)
        if failed == 0:
            print(f"✨ All {passed}/{len(results)} structured output examples completed!")
        else:
            print(
                f"⚠️ {passed}/{len(results)} structured output examples completed; {failed} failed"
            )
            for name, ok in results:
                status = "✓" if ok else "✗"
                print(f"   {status} {name}")

        print("\n💡 Key Takeaways:")
        print("   • Define schemas as Pydantic BaseModel classes")
        print(
            "   • chat.completions.parse(response_format=ModelClass) returns ParsedChatCompletion[T]"
        )
        print("   • parsed.parsed is a typed instance — no manual json.loads needed")
        print("   • Pydantic raises ValidationError on schema violations")
        print("   • Lower temperature values improve consistency")
        print("\n📚 Learn more at: https://docs.venice.ai/structured-outputs")

        return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    sys.exit(exit_code)
