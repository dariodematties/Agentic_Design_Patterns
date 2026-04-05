from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch

#--- Configuration ---
# Ensure your API key is set in the environment variable 'OPENAI_API_KEY'
try:
  llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
  print(f"LLM {llm.model_name} initialized successfully.")
except Exception as e:
  print(f"Error initializing LLM: {e}")
  llm = None

#--- Define Simulated Sub-Agent Handlers (equivalent to ADK sub-agents) ---

def booking_handler(request:str) -> str:
    # Simulates the Booking Agent handling a booking request
    print("\n--- DELLEGATING TO BOOKING HANDLER ---")
    return f"Booking Handler processed request: '{request}'.Result: Simulated booking action completed."

def info_handler(request:str) -> str:
    # Simulates the Information Agent handling an information request
    print("\n--- DELLEGATING TO INFO HANDLER ---")
    return f"Info Handler processed request: '{request}'.Result: Simulated information retrieval completed."

def unclear_handler(request:str) -> str:
    # Handles requests that could't be delegated
    print("\n--- HANDLING UNCLEAR REQUEST ---")
    return f"Coordinator could not delegate request: '{request}'. Please clarify your request."


# --- Define the Coordinator Router Chain (equivalent to ADK Coordinator's instruction) ---
# This chain decides which handler to delegate to based on the content of the request.
coordinator_router_prompt = ChatPromptTemplate.from_messages([
    ("system", """Analyze the user's request and determine which handler should process it.
    - If the request is related to booking (e.g., "book a flight", "reserve a hotel"), 
         output 'booking'.
     - For all other information questions (e.g., "what is the weather?", "who is the president?"), output 'info'.
     - If the request is unclear or cannot be categorized, output 'unclear'.
     ONLY output one of the following: 'booking', 'info', or 'unclear'."""),
    ("human", "{request}")
])


if llm:
    # --- Build the Coordinator Router Chain ---
    coordinator_router_chain = coordinator_router_prompt | llm | StrOutputParser()

# --- Define the Delegation Logic (equivalent to ADK's Auto-Flow based on sub-agent capabilities) ---
# Use RunnableBranch to route based on the router chain's output.

# Define the branching logic for the RunnableBranch
branches = {
        "booking": RunnablePassthrough.assign(output=lambda x: booking_handler(x["request"]['request'])),
        "info": RunnablePassthrough.assign(output=lambda x: info_handler(x["request"]['request'])),
        "unclear": RunnablePassthrough.assign(output=lambda x: unclear_handler(x["request"]['request']))
}



# Create the RunnableBranch with the defined branches. It takes the output of the router chain and routes the original input ('request') to the appropriate handler.
delegation_branch = RunnableBranch(
        (lambda x: x['decision'].strip() == 'booking', branches['booking']), # Added .strip() to clean up any whitespace from the LLM output
        (lambda x: x['decision'].strip() == 'info', branches['info']),
        branches['unclear'] # Default branch if no conditions are met
)

# Combine the router chanin and the delegation branch into a single runable.
# The router chain's output ('decision') is passed along with the original input ('request') to the delegation_branch.
coordinator_agent = {
        "decision": coordinator_router_chain, # This will produce the 'decision' key in the input for the delegation branch
        "request": RunnablePassthrough() # This will pass the original request through to the delegation branch
} | delegation_branch | (lambda x: x['output']) # Extract the final output from the delegation branch


# --- Example Usage ---
def main():
    if not llm:
        print("LLM is not initialized. Exiting.")
        return

    print("\n--- Running with a booking request ---")
    request_a = "Book me a flight to Paris."
    result_a = coordinator_agent.invoke({"request": request_a})
    print(f"Final Result: {result_a}")

    print("\n--- Running with an information request ---")
    request_b = "What is the capital of France?"
    result_b = coordinator_agent.invoke({"request": request_b})
    print(f"Final Result: {result_b}")

    print("\n--- Running with an unclear request ---")
    request_c = "I want to do something fun."
    result_c = coordinator_agent.invoke({"request": request_c})
    print(f"Final Result: {result_c}")

if __name__ == "__main__":
    main()


