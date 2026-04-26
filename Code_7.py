import asyncio
import uuid

from google.adk.agents import SequentialAgent, LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types


APP_NAME = "reflection_review_pipeline"
USER_ID = "user_123"


# The first agent generates the initial draft
generator_agent = LlmAgent(
        name="DraftWriter",
        description="Generates initial draft content on a given subject.",
        instruction="Write a short informative paragraph about the user's subject.",
        output_key="draft_text" # The output is saved in this state key for the next agent to use.
)


# The second agent critiques the draft from the first agent
reviewer_agent = LlmAgent(
        name="FactChecker",
        description="Reviews a given text for actual accuracy and provides a structured critique.",
        instruction="""
        You are a meticulous fact-checker.
        1. Read the text provided in the state key 'draft_text'.
        2. Carefully verify the factual accuracy of all claims made in the text.
        3. Your final output must be a dictionary containing two keys:
          - 'status': A string that is either 'ACCURATE' if all facts are correct, or 'INACCURATE' if any factual errors are found.
          - 'reasoning': A string providing a clear explanation of your status, citing specific issues if any are found.
        """,
        output_key="review_output" # The structured dictionary is saved here.
)

# The SequentialAgent ensures the generator runs before the reviewer.
review_pipeline = SequentialAgent(
        name="WriteAndReviewPipeline",
        sub_agents=[generator_agent, reviewer_agent]
)

# Execution Flow:
# 1. generator_agent runs -> saves its paragraph to state['draft_text']
# 2. reviewer_agent runs -> reads state['draft_text'] and saves its dictionary output to state['review_output']

root_agent = review_pipeline


async def run_pipeline(runner: InMemoryRunner, request: str) -> str:
    """Run the review pipeline and return the final text output."""
    session_id = str(uuid.uuid4())
    final_result = ""

    await runner.session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=request)],
        ),
    ):
        if event.is_final_response() and event.content:
            if getattr(event.content, "text", None):
                final_result = event.content.text
            elif getattr(event.content, "parts", None):
                text_parts = [part.text for part in event.content.parts if getattr(part, "text", None)]
                final_result = "".join(text_parts)

    return final_result


async def main() -> None:
    """Run the ADK draft-and-review example."""
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    request = (
        "Write a short informative paragraph about the current state of "
        "quantum computing, including its practical applications, major "
        "technical limitations, and whether it is likely to replace "
        "classical computing in the near future."
    )
    result = await run_pipeline(runner, request)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
