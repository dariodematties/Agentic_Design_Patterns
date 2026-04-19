import os
import asyncio
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough

#--- Configuration ---
# Ensure your API key is set in the environment variable is set correctly. (e.q "OPENAI_API_KEY")

try:
    llm: Optional[ChatOpenAI] = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
        )
    print("ChatOpenAI initialized successfully.")

except Exception as e:
    print(f"Error initializing ChatOpenAI: {e}")
    llm = None


#--- Define Independent Chains ---
# These three chains represent distinct tasks that can be executed in parallel.

summarize_chain: Runnable = (ChatPromptTemplate.from_messages([
    ("system", "Summarize the following topic in a concise manner."),
    ("user", "{input}")
    ])
    | llm
    | StrOutputParser()
)

question_chain: Runnable = (ChatPromptTemplate.from_messages([
    ("system", "Generate three interesting questions about the following topic."),
    ("user", "{input}")
    ])
    | llm
    | StrOutputParser()
)

term_chain: Runnable = (ChatPromptTemplate.from_messages([
    ("system", "Identify 5-10 key terms related to the following topic, separated by commas."),
    ("user", "{input}")
    ])
    | llm
    | StrOutputParser()
)

#--- Build the Parallel + Synthesis Chain ---

# 1. define the block of chains to run in parallel. The result of these
# along with the original topic will be fed into the next step for synthesis.
map_chain = RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": question_chain,
            "key_terms": term_chain,
            "topic": RunnablePassthrough(),  # Pass the original input through to the next step
        }
)

# 2. Define the final synthesis prompt which will combine the outputs from the parallel chains into a cohesive response.
synthesis_prompt = ChatPromptTemplate.from_messages([
    ("system", """Based on the following information:
     Summary: {summary}
     Related Questions: {questions}
     key Terms: {key_terms}
     Synthesize a comprehensive answer."""),
    ("user", "Original Topic: {topic}")
    ])


# 3. Construct the full chain by piping the parallel results directly into the synthesis prompt followed by the LLM output parser.
full_parallel_chain = map_chain | synthesis_prompt | llm | StrOutputParser()

#---Run the Chain ---
async def run_parallel_example(topic: str) -> None:
    """
    Asyynchronously invokes the full parallel processing chain with a specific topic
    and prints the final synthesized output.
    
    Args:
        topic (str): The input topic to be processed by the LangChain chain.
    """
    if llm is None:
        print("LLM is not initialized. Please check your API key and configuration.")
        return

    print(f"\n--- Running Parallel LangChain Example for Topic: '{topic}' ---")
    try:
        # The input to 'ainvoke' is the single topic string,
        # then passed to each runnable in the 'map_chain'.
        response = await full_parallel_chain.ainvoke(topic)
        print(f"\n--- Final Synthesized Response ---\n{response}")
    except Exception as e:
        print(f"An error occurred while running the parallel chain: {e}")



if __name__ == "__main__":
    test_topic = "The history of space exploration"
    # In python 3.7+, you can use asyncio.run() to execute the async function.
    asyncio.run(run_parallel_example(test_topic))




