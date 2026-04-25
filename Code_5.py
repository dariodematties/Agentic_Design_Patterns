import asyncio
import uuid

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types

GEMINI_MODEL = "gemini-2.5-flash"
APP_NAME = "parallel_research_pipeline"
USER_ID = "user_123"


# --- Define Researcher Sub-Agents (to run in parallel) ---

# Researcher 1: Renewable Energy Expert
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Reseach Assistance specializing in energy.
    Research the latest advancements in 'renewable energy sources'.
    Use the Google Search tool provided.
    Summarize your key findings in a concise manner.
    Output *only* the summary.
    """,
    description="Researches renewable energy sourves.",
    tools=[google_search],
    # Store results in state for merger agent
    output_key="renewable_energy_result"
    )

# Researcher 2: Electric Vehicle Expert
researcher_agent_2 = LlmAgent(
    name="ElectricVehicleResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Reseach Assistance specializing in transportation.
    Research the latest advancements in 'electric vehicles technology'.
    Use the Google Search tool provided.
    Summarize your key findings in a concise manner.
    Output *only* the summary.
    """,
    description="Researches electric vehicle technology.",
    tools=[google_search],
    # Store results in state for merger agent
    output_key="electric_vehicle_result"
    )

# Researcher 3: Carbon Capture Expert
researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=GEMINI_MODEL,
    instruction="""You are an AI Reseach Assistance specializing in climate solutions.
    Research the current state of 'carbon capture technologies'.
    Use the Google Search tool provided.
    Summarize your key findings in a concise manner.
    Output *only* the summary.
    """,
    description="Researches carbon capture technologies.",
    tools=[google_search],
    # Store results in state for merger agent
    output_key="carbon_capture_result"
    )


# --- Create the ParallelAgent to run all researchers concurrently ---
# This agent orchestrates the execution of all three researcher agents in parallel.
# It finishes once all researchers have completed their tasks and stored their results in the shared state.
parallel_research_agent = ParallelAgent(
        name="ParallelWebResearcherAgent",
        sub_agents=[researcher_agent_1, researcher_agent_2, researcher_agent_3],
        description="Runs multiple research agents in parallel to gather information."
        )




# --- Define the Merger Agent (Runs after all parallel agents complete) ---
# This agent takes the results stored in the session state by the parallel agents.
# and synthesizes them into a comprehensive report with attributions.
merger_agent = LlmAgent(
    name="SynthesisAgent",
    model=GEMINI_MODEL, # Or a more powerful model if available for better synthesis
    instruction="""You are an AI assistant responsible for combining the research findings from multiple experts into a cohesive report.
    Your primary task is to synthesize the following research summaries, clearly attributing each piece of information to its respective source.
    Structure your response using headings for each topic, and ensure that the information is presented in a clear and organized manner.

    ** Crucially: Your entire report MUST be grounded *exclusively* in the information provided in the input summeries below.
    Do NOT include any external knowledge, facts, details not present in the summaries.**

    **Input Summaries:**
    - Renewable Energy Summary: {renewable_energy_result}
    - Electric Vehicle Summary: {electric_vehicle_result}
    - Carbon Capture Summary: {carbon_capture_result}

    **Output Format:**

    ## Summary of Recent Sustainable Technology Advancements 

    ### Renewable Energy Findings
    (Based on RenewableEnergyResearcher's findings)
    [Synthetize and elaborate *only* on the renewable energy information provided in the summary above.]

    ### Electric Vehicle Findings
    (Based on ElectricVehicleResearcher's findings)
    [Synthetize and elaborate *only* on the electric vehicle information provided in the summary above.]

    ### Carbon Capture Findings
    (Based on CarbonCaptureResearcher's findings)
    [Synthetize and elaborate *only* on the carbon capture information provided in the summary above.]

    ### Overall Conclusion
    [Provide a brief (1-2 sentences) conclusion that connects *only* the findings from the three summaries above.]

    Output *only* the structured report following this format. Do not include introductory or concluding remarks outside of the specified sections, and strictly adehere to using only the information provided in the input summaries.
    """,
    description="Combines research findings from parallel agents into a structured report with clear attributions, strictly grounded in the provided summaries.",
    # No tools needed for merging
    # No output key needed since the direct response is the final output of the sequence
    )


# --- Create the SequentialAgent to orchestrate the entire workflow ---
# This is the main agent that will be run. It first runs the parallel research agent (ParallelAgent)
# to populate the state, and then executes the merger agent (SynthesisAgent) to produce the final output.
sequential_pipeline_agent = SequentialAgent(
    name="ResarchAndSynthesisPipelineAgent",
    # Run parallel research first, then synthesis
    sub_agents=[parallel_research_agent, merger_agent],
    description="Coordinates parallel research and synthesizes the results into a comprehensive report."
    )

root_agent = sequential_pipeline_agent


async def run_pipeline(runner: InMemoryRunner, request: str) -> str:
    """Run the sequential pipeline and return the final text output."""
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
    runner = InMemoryRunner(agent=root_agent, app_name=APP_NAME)
    request = (
        "Research the latest advancements in renewable energy, electric vehicle technology, "
        "and carbon capture technologies, then produce a structured synthesis."
    )
    result = await run_pipeline(runner, request)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
