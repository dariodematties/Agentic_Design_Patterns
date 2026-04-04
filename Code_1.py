import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize the language model
llm = ChatOpenAI(temperature=0)

# ---Prompt 1: Extract Information---
prompt_extract = ChatPromptTemplate.from_template(
        "Extract the technical specification from the following text:\n\n{text_input}"
)

# ---Prompt 2: Transform to JSON---
prompt_transform = ChatPromptTemplate.from_template(
        "Transform the following technical specification into a JSON object with 'cpu', 'memory', and 'storage' fields:\n\n{specifications}"
)

# ---Build the chain using LCEL---
# The StrOutputParser() converts the LLM's output into a string format that can be easily used in the next step.
extraction_chain = prompt_extract | llm | StrOutputParser()

# The full chain passes the output of the extraction chain into the 'specifications' 
# variable of the transformation prompt
full_chain = (
        {"specifications": extraction_chain} | prompt_transform | llm | StrOutputParser()
)

# Example input text
# input_text = "The new laptop features an Intel i7 processor, 16GB of RAM, and a 512GB SSD."
# input_text = "The server has an AMD Ryzen 9 CPU, 32GB of memory, and a 1TB NVMe SSD."
input_text = "The desktop computer is equipped with an Intel Core i5 processor, 8GB of RAM, and a 256GB SSD."

# Run the full chain
output = full_chain.invoke({"text_input": input_text})

print("Extracted and Transformed Output:")
print(output)

