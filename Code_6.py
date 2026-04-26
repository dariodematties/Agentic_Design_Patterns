import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# ---Configuration---
# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()

# Check if the API key is set
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set in the environment variables.")
else:
    print("OPENAI_API_KEY is set.")

# Initialize the Chat LLM. We use gpt-4o for better reasoning.
# A lower temperature is use for more deterministic outputs, which is often desirable in structured tasks.
try:
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )
    print("ChatOpenAI initialized successfully.")
except Exception as e:
    print(f"Error initializing ChatOpenAI: {e}")
    llm = None


def run_reflection_loop():
    """
    Demonstrates a multi-step AI reflection loop to progressively improve a Python function.
    """
    #--- The Core Task ---
    task_prompt = """
    Your task is to write a Python function named `calculate_factorial`.
    This function should do the following:
    1. Take a single integer input `n`.
    2. Calculate the factorial of `n` (i.e., n! = n * (n-1) * ... * 1, with 0! = 1).
    3. Include a clear docstring explaining the function's purpose, parameters, and return value, as well as what the function does.
    4. Handle edge cases (e.g. the factorial of 0 is 1).
    5. Handle invalid input: Raise a ValueError if the input is negative or not an integer.
    """

    #--- The Reflection Loop ---
    max_iterations = 10  # Maximum number of iterations to prevent infinite loops in case of issues.
    current_code = ""
    # We will build a conversation history to provide context in each step.
    message_history = [HumanMessage(content=task_prompt)]

    for iteration in range(max_iterations):
        print("\n" + "="*25 + f" REFLECTION LOOP ITERATION {iteration + 1} " + "="*25)

        # --- 1. GENERATE / REFINE STAGE ---
        # In the first iteration, the model will generate the initial code. In subsequent iterations, it will refine the code based on feedback.
        if iteration == 0:
            print("\n>>> STAGE 1: GENERATING initial code...")
            # The first message is just the task prompt
            response = llm.invoke(message_history)
            current_code = response.content
        else:
            print("\n>>> STAGE 1: REFINING code based on feedback...")
            # The message history now includes the task prompt, the previous code, and the feedback.
            # We ask the model to refine the code based on the feedback.
            message_history.append(HumanMessage(content="Please refine the code based on the feedback provided."))
            response = llm.invoke(message_history)
            current_code = response.content

        print("\nGenerated/Refined Code (v" + str(iteration + 1) + "):\n" + current_code)
        message_history.append(response) # Add the generated code history.

        # --- 2. REFLECT STAGE ---
        print("\n>>> STAGE 2: REFLECTING on the generated code...")

        # Create a specific prompt for the reflector agent.
        # This ask the model to act as a senior code reviewer and provide detailed feedback on the code.
        reflector_prompt = [
                SystemMessage(content="""
                              You are a senior software engineer and an expert in Python.
                              Your role is to perform a meticulous code review of the provided `calculate_factorial` function.
                              Critically evaluate the provided Python code based on the original task requirements.
                              Look for bugs, style issues, missing edge cases, and areas for improvement.
                              If the code is perfect and meets all requirements,
                              respond with the single phrase 'CODE_IS_PERFECT'.
                              Otherwise, provide a bulleted list of specific, actionable 
                              feedback points that the code author can use to improve the code in the next iteration.
                              """),
                HumanMessage(content=f"Original Task:\n{task_prompt}\n\nCode to Review:\n{current_code}")
                ]

        critique_response = llm.invoke(reflector_prompt)
        critique = critique_response.content

        #--- 3. STOPPING CONDITION CHECK ---
        if "CODE_IS_PERFECT" in critique:
            print("\n--- Critique ---\nNo further critiques found. The code is satisfactory.")
            break

        print("\n--- Critique ---\n" + critique) 
        # Add the critique to the history of the next refinement loop.
        message_history.append(HumanMessage(content=f"Critique of the previous code:\n{critique}"))

    print("\n" + "="*30 + " FINAL CODE AFTER REFLECTION LOOP " + "="*30)
    print("\nFinal refined code after the refletion process:\n" + current_code)

if __name__ == "__main__":
    if llm is not None:
        run_reflection_loop()
    else:
        print("LLM is not initialized. Please check your configuration and try again.")



