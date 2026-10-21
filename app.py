import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set."
    )


# ============================================================
# 2. GEMMA AGENT CORE
# ============================================================

llm_flash = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 3. RESEARCH ANALYSIS TOOL
# ============================================================

@tool
def research_analysis(
    research_topic: str,
    research_question: str,
    research_goal: str
) -> str:
    """
    Analyze a research topic, question and goal
    and generate a structured research analysis.
    """

    prompt = f"""

You are an expert academic research assistant.

RESEARCH TOPIC:
{research_topic}

RESEARCH QUESTION:
{research_question}

RESEARCH GOAL:
{research_goal}

Analyze the research problem carefully.

Provide the following:

1. RESEARCH PROBLEM
Clearly explain the problem being studied.

2. EXISTING KNOWLEDGE
Explain the general existing approaches,
methods or technologies related to the topic.

3. RESEARCH GAP
Identify limitations, challenges or areas
that can be improved.

4. PROPOSED APPROACH
Suggest a practical methodology or approach
that could address the research gap.

5. EXPECTED OUTCOME
Explain what the research is expected to achieve.

6. FINAL RESEARCH SUMMARY
Give a short summary connecting the topic,
question, goal, gap and proposed approach.

IMPORTANT:

- Do not invent specific research papers,
authors, statistics or experimental results.
- Do not claim that something is proven unless
  it is clearly supported by the information given.
- Use general academic knowledge.
- Keep the explanation understandable to a
  college student.
- Return normal readable text.
- Do not return JSON.
- Do not return Python dictionaries.
- Do not use Markdown code fences.
"""

    try:

        response = llm_flash.invoke(
            prompt
        )

        content = response.content

        if isinstance(
            content,
            str
        ):

            return content.strip()

        return str(content)

    except Exception as e:

        return (
            "Research analysis failed: "
            + str(e)
        )


# ============================================================
# 4. TOOLS AVAILABLE TO GEMMA
# ============================================================

tools = [
    research_analysis
]


# ============================================================
# 5. RESEARCHMATE AI AGENT
# ============================================================

system_prompt = """

You are ResearchMate AI.

You are an AI research assistant designed to
help college students understand and develop
research ideas.

The student provides only:

1. Research Topic
2. Research Question
3. Research Goal


YOUR WORKFLOW:

STEP 1:

Understand the research topic.

STEP 2:

Understand the research question.

STEP 3:

Understand the research goal.

STEP 4:

Use the Research Analysis tool to analyze
the research idea.

STEP 5:

Present the final research analysis in a
simple academic format.


IMPORTANT:

Do not ask the student for additional information.

Do not expose internal reasoning.

Do not expose tool calls.

Do not output JSON.

Do not output Python dictionaries.

Do not invent research papers, authors,
statistics or experimental results.


FINAL RESPONSE FORMAT:

RESEARCHMATE AI REPORT

Research Topic:

Research Question:

Research Goal:

Research Problem:

Existing Knowledge:

Research Gap:

Proposed Approach:

Expected Outcome:

Final Research Summary:


The explanation must be simple enough for
a college student to understand.

Use clear paragraphs and bullet points where
useful.
"""


# ============================================================
# 6. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=system_prompt

)


# ============================================================
# 7. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    research_topic: str = Field(
        description=(
            "The research topic"
        )
    )

    research_question: str = Field(
        description=(
            "The main research question"
        )
    )

    research_goal: str = Field(
        description=(
            "The main goal of the research"
        )
    )


# ============================================================
# 8. FORMAT INPUT FOR GEMMA AGENT
# ============================================================

def format_for_agent(x) -> dict:

    if isinstance(x, dict):

        research_topic = x.get(
            "research_topic",
            ""
        )

        research_question = x.get(
            "research_question",
            ""
        )

        research_goal = x.get(
            "research_goal",
            ""
        )

    else:

        research_topic = getattr(
            x,
            "research_topic",
            ""
        )

        research_question = getattr(
            x,
            "research_question",
            ""
        )

        research_goal = getattr(
            x,
            "research_goal",
            ""
        )


    user_input = f"""

RESEARCHMATE AI INPUT

==================================================
RESEARCH TOPIC
==================================================

{research_topic}


==================================================
RESEARCH QUESTION
==================================================

{research_question}


==================================================
RESEARCH GOAL
==================================================

{research_goal}


==================================================
TASK
==================================================

Analyze this research idea.

Identify:

1. Research Problem
2. Existing Knowledge
3. Research Gap
4. Proposed Approach
5. Expected Outcome
6. Final Research Summary

Return the final research report as
clear readable text.
"""


    return {

        "messages": [

            (
                "user",
                user_input
            )

        ]

    }


# ============================================================
# 9. EXTRACT FINAL TEXT
# ============================================================

def extract_text_response(
    agent_output
) -> str:

    if isinstance(
        agent_output,
        str
    ):

        return agent_output.strip()


    if not isinstance(
        agent_output,
        dict
    ):

        return str(agent_output)


    messages = agent_output.get(
        "messages"
    )


    if messages is None:

        for value in agent_output.values():

            if (
                isinstance(
                    value,
                    dict
                )
                and
                "messages" in value
            ):

                messages = value[
                    "messages"
                ]

                break


    if not messages:

        return (
            "No final research report "
            "was generated."
        )


    for message in reversed(
        messages
    ):

        if message.__class__.__name__ != "AIMessage":

            continue


        content = getattr(
            message,
            "content",
            ""
        )


        if isinstance(
            content,
            str
        ):

            text = content.strip()

            if text:

                return text


        if isinstance(
            content,
            list
        ):

            text_parts = []


            for item in content:

                if isinstance(
                    item,
                    dict
                ):

                    item_type = item.get(
                        "type",
                        ""
                    )


                    if item_type in {
                        "thinking",
                        "reasoning"
                    }:

                        continue


                    if item_type in {
                        "text",
                        "output_text"
                    }:

                        text = item.get(
                            "text",
                            ""
                        )

                        if text:

                            text_parts.append(
                                text
                            )

                elif isinstance(
                    item,
                    str
                ):

                    text_parts.append(
                        item
                    )


            final_text = "\n".join(
                text_parts
            ).strip()


            if final_text:

                return final_text


    return (
        "No readable research report "
        "was generated."
    )


# ============================================================
# 10. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (

    RunnableLambda(
        format_for_agent
    )

    | agent

    | RunnableLambda(
        extract_text_response
    )

).with_types(

    input_type=AgentInput,

    output_type=str

)


# ============================================================
# 11. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "ResearchMate AI API"
    ),

    version="1.0",

    description=(
        "ResearchMate AI using "
        "LangChain, Gemma-4-31B-IT, "
        "FastAPI and LangServe."
    )

)


# ============================================================
# 12. LANGSERVE ROUTER
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/research-agent"

)


# ============================================================
# 13. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "ResearchMate AI is running.",

        "agent_core":
            "gemma-4-31b-it",

        "endpoint":
            "/research-agent",

        "playground":
            "/research-agent/playground/"

    }


# ============================================================
# 14. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            8000
        )

    )

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=port

    )
