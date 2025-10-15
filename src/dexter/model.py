import os
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Type, List, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import AIMessage

from dexter.prompts import DEFAULT_SYSTEM_PROMPT

# Initialize the Anthropic client
# Make sure your ANTHROPIC_API_KEY is set in your environment
llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0, api_key=os.getenv("ANTHROPIC_API_KEY"))

def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
) -> AIMessage:
  final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
  
  prompt_template = ChatPromptTemplate.from_messages([
      ("system", final_system_prompt),
      ("user", "{prompt}")
  ])

  runnable = llm
  if output_schema:
      runnable = llm.with_structured_output(output_schema)
  elif tools:
      runnable = llm.bind_tools(tools)
  
  chain = prompt_template | runnable
  return chain.invoke({"prompt": prompt})
