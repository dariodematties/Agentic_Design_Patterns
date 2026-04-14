import asyncio
import uuid

from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import FunctionTool
from google.genai import types


APP_NAME = "agentic_design_patterns"
USER_ID = "user_123"


# --- Define Tool Functions ---
# These functions simulate the actions of the specialist agents.

def booking_handler(request: str) -> str:
    """Handle booking-related requests."""
    print("\n--- DELLEGATING TO BOOKING HANDLER ---")
    return (
        f"Booking Handler processed request: '{request}'. "
        "Result: Simulated booking action completed."
    )


def info_handler(request: str) -> str:
    """Handle general information requests."""
    print("\n--- DELLEGATING TO INFO HANDLER ---")
    return (
        f"Info Handler processed request: '{request}'. "
        "Result: Simulated information retrieval completed."
    )


def unclear_handler(request: str) -> str:
    """Handle requests that are unclear or cannot be categorized."""
    return f"Coordinator could not delegate request: '{request}'. Please clarify your request."


# --- Create Tools from Functions ---
booking_tool = FunctionTool(booking_handler)
info_tool = FunctionTool(info_handler)


# Define specialized sub-agents equipped with their respective tools.
booking_agent = Agent(
    name="Booker",
    model="gemini-2.5-flash",
    description=(
        "A specialized agent that handles flight and hotel booking requests. "
        "Use the booking_tool to process booking tasks."
    ),
    tools=[booking_tool],
)

info_agent = Agent(
    name="Info",
    model="gemini-2.5-flash",
    description=(
        "A specialized agent that handles general information requests. "
        "Use the info_tool to process information tasks."
    ),
    tools=[info_tool],
)


# --- Define the parent agent with explicit delegation logic ---
coordinator = Agent(
    name="Coordinator",
    model="gemini-2.5-flash",
    instruction=(
        "You are a coordinator agent that receives user requests and delegates them "
        "to the appropriate specialist agent based on the content of the request. "
        "Do not attempt to answer the request yourself. Instead, analyze the request "
        "and decide which agent should handle it:\n"
        "- If the request is related to booking (for example, 'book a flight' or "
        "'reserve a hotel'), delegate to the Booker agent.\n"
        "- For all other information questions (for example, 'what is the weather?' "
        "or 'who is the president?'), delegate to the Info agent."
    ),
    description=(
        "A coordinator agent that routes user requests to the appropriate "
        "specialist agent."
    ),
    sub_agents=[booking_agent, info_agent],
)


# --- Execution Logic ---
async def run_coordinator(runner: InMemoryRunner, request: str) -> str:
    """Run the coordinator agent with a given request."""
    print(f"\n--- Running Coordinator with request: '{request}' ---")
    final_result = ""

    try:
        session_id = str(uuid.uuid4())

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
                    text_parts = [part.text for part in event.content.parts if part.text]
                    final_result = "".join(text_parts)
                break

        print(f"\n--- Final Result from Coordinator: {final_result} ---")
        return final_result

    except Exception as e:
        print(f"An error occurred while processing your request: {e}")
        return f"An error occurred while processing your request: {e}"


async def main() -> None:
    """Run the ADK example."""
    print("\n--- Google ADK Routing Example (ADK Auto-Flow Style) ---")
    print("Note: This requires Google ADK installed and authenticated.")

    runner = InMemoryRunner(agent=coordinator, app_name=APP_NAME)

    result_a = await run_coordinator(
        runner,
        "Book me a hotel in New York for next weekend.",
    )
    print(f"\nFinal Output A: {result_a}")

    result_b = await run_coordinator(
        runner,
        "What is the highest mountain in the world?",
    )
    print(f"\nFinal Output B: {result_b}")

    result_c = await run_coordinator(
        runner,
        "Tell me a random fact.",
    )
    print(f"\nFinal Output C: {result_c}")

    result_d = await run_coordinator(
        runner,
        "Find flights to Tokyo next month.",
    )
    print(f"\nFinal Output D: {result_d}")


if __name__ == "__main__":
    asyncio.run(main())
