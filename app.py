import os
import uvicorn
import requests

from fastapi import FastAPI
from langserve import add_routes

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

from pydantic import BaseModel, Field
from ddgs import DDGS


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
# 3. RESEARCH SEARCH TOOL
# ============================================================

@tool
def research_search(
    research_topic: str
) -> str:
    """
    Search for research papers, articles,
    datasets and technical information related
    to the research topic.
    """

    query = (
        f"{research_topic} research paper "
        f"methodology dataset recent findings"
    )

    try:

        results = DDGS().text(
            query,
            max_results=7
        )

        if not results:
            return "No research information was found."

        output = []

        for item in results:

            title = item.get(
                "title",
                "Research Source"
            )

            url = item.get(
                "href",
                ""
            )

            description = item.get(
                "body",
                ""
            )

            output.append(
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Description: {description}"
            )

        return "\n\n".join(output)

    except Exception as e:

        return (
            "Research search failed: "
            + str(e)
        )


# ============================================================
# 4. RESEARCH ANALYSIS TOOL
# ============================================================

@tool
def research_analysis(
    research_topic: str,
    research_sources: str
) -> str:
    """
    Analyze the collected research information
    and identify important findings, methods,
    datasets and limitations.
    """

    prompt = f"""

You are a research analysis assistant.

RESEARCH TOPIC:
{research_topic}

RESEARCH SOURCES:
{research_sources}

Analyze the available research information.

Identify:

1. Research problem
2. Important findings
3. Common methodologies
4. Models or techniques used
5. Datasets mentioned
6. Important advantages
7. Important limitations
8. Research trends

IMPORTANT:

Only use information supported by the
provided research sources.

Do not invent papers, results, datasets
or research findings.

Clearly mention when information is not
available.

Return clear normal text.
Do not return JSON.
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
# 5. RESEARCH COMPARISON TOOL
# ============================================================

@tool
def research_comparison(
    research_topic: str,
    research_analysis_text: str
) -> str:
    """
    Compare the important research approaches
    and techniques identified during analysis.
    """

    prompt = f"""

You are a research comparison assistant.

RESEARCH TOPIC:
{research_topic}

RESEARCH ANALYSIS:
{research_analysis_text}

Compare the important approaches, models
or techniques found in the research.

For each important approach provide:

Approach/Model:
Purpose:
Main Advantage:
Main Limitation:
Dataset or Data Used:
Performance Information:
Suitability for Future Research:

Then provide:

Best Approach:
Reason:

IMPORTANT:

Do not invent numerical performance values.

If performance information is unavailable,
clearly say so.

Return normal readable text.
Do not return JSON.
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
            "Research comparison failed: "
            + str(e)
        )


# ============================================================
# 6. RESEARCH GAP TOOL
# ============================================================

@tool
def research_gap(
    research_topic: str,
    research_analysis_text: str,
    comparison_text: str
) -> str:
    """
    Identify possible research gaps and
    opportunities for future research.
    """

    prompt = f"""

You are a research gap identification assistant.

RESEARCH TOPIC:
{research_topic}

RESEARCH ANALYSIS:
{research_analysis_text}

RESEARCH COMPARISON:
{comparison_text}

Identify realistic research gaps.

Analyze:

1. Limitations in existing approaches
2. Problems that are not fully solved
3. Dataset-related limitations
4. Model-related limitations
5. Performance-related limitations
6. Real-world implementation limitations
7. Possible improvement areas
8. Potential future research directions

IMPORTANT:

Do not claim that a topic is completely
unexplored unless the provided information
supports that conclusion.

Use phrases such as:

"Possible research gap"
"Potential improvement area"
"Further investigation may be required"

Return clear normal text.
Do not return JSON.
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
            "Research-gap analysis failed: "
            + str(e)
        )


# ============================================================
# 7. METHODOLOGY RECOMMENDATION TOOL
# ============================================================

@tool
def methodology_recommendation(
    research_topic: str,
    research_gap_text: str
) -> str:
    """
    Suggest a practical research methodology
    based on the identified research gaps.
    """

    prompt = f"""

You are a research methodology advisor.

RESEARCH TOPIC:
{research_topic}

IDENTIFIED RESEARCH GAPS:
{research_gap_text}

Suggest a practical methodology for a
student-level research project.

Include:

1. Problem Definition
2. Data Collection
3. Dataset Preparation
4. Proposed Technique or Model
5. Training or Implementation Steps
6. Evaluation Metrics
7. Comparison with Existing Methods
8. Expected Outcome
9. Possible Challenges

The methodology must be realistic for a
college research project.

Do not invent experimental results.

Return normal readable text.
Do not return JSON.
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
            "Methodology recommendation failed: "
            + str(e)
        )


# ============================================================
# 8. TOOLS AVAILABLE TO GEMMA
# ============================================================

tools = [
    research_search,
    research_analysis,
    research_comparison,
    research_gap,
    methodology_recommendation
]


# ============================================================
# 9. RESEARCHMATE AI AGENT
# ============================================================

system_prompt = """

You are ResearchMate AI.

Your purpose is to help college students,
researchers and beginners understand
research topics and plan practical research.

The student provides:

1. Research Topic
2. Research Question
3. Research Goal


YOUR WORKFLOW:

STEP 1:
Analyze the research topic and use the
Research Search tool to find relevant
research information.

STEP 2:
Use the Research Analysis tool to analyze
the collected research information.

STEP 3:
Use the Research Comparison tool to compare
important models, techniques or approaches.

STEP 4:
Use the Research Gap tool to identify
possible research gaps and improvement areas.

STEP 5:
Use the Methodology Recommendation tool
to suggest a practical research methodology.

STEP 6:
Combine all results and produce the final
ResearchMate AI Report.


IMPORTANT:

YOU ARE THE AGENT CORE.

You must decide when a tool is required.

Use the tools available to you.

Do not invent research papers.

Do not invent citations, datasets,
performance values or experimental results.

Do not assume that a tool was successful if
it returned an error.

Do not expose internal reasoning.

Do not expose tool calls.

Do not output JSON.

Do not output Python dictionaries.

Do not output Markdown code blocks.


FINAL RESPONSE FORMAT:

RESEARCHMATE AI REPORT

Research Topic:

Research Question:

Research Overview:

Relevant Research:

Important Findings:

Research Methods:

Model/Technique Comparison:

Datasets:

Possible Research Gaps:

Suggested Methodology:

Research Direction:

Sources:


The final response must be understandable
to a college student.

Use paragraphs and bullet points where useful.

The final response must contain actual information
obtained from the research search, research analysis,
comparison, research-gap analysis and methodology
recommendation.
"""


# ============================================================
# 10. CREATE LANGCHAIN AGENT
# ============================================================

agent = create_agent(

    model=llm_flash,

    tools=tools,

    system_prompt=system_prompt

)


# ============================================================
# 11. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    research_topic: str = Field(
        description=(
            "Research topic that the student "
            "wants to investigate"
        )
    )

    research_question: str = Field(
        description=(
            "Main research question that the "
            "student wants to investigate"
        )
    )

    research_goal: str = Field(
        description=(
            "Goal or objective of the research"
        )
    )


# ============================================================
# 12. FORMAT INPUT FOR GEMMA AGENT
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

Analyze this research topic.

You are the Agent Core.

Select and use the appropriate tools.

Follow the complete workflow:

Research Search
        ↓
Research Analysis
        ↓
Research Comparison
        ↓
Research Gap Identification
        ↓
Methodology Recommendation
        ↓
Final Research Synthesis

Return the final ResearchMate AI report
as readable text.
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
# 13. EXTRACT FINAL TEXT
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

        # Ignore tool messages
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


                    # Remove Gemini thinking/reasoning
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
        "No final readable research "
        "report was generated."
    )


# ============================================================
# 14. LANGSERVE CHAIN
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
# 15. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title=(
        "ResearchMate AI API"
    ),

    version="1.0",

    description=(
        "ResearchMate AI using "
        "LangChain, Gemma-4-31B-IT, "
        "DuckDuckGo, FastAPI "
        "and LangServe."
    )

)


# ============================================================
# 16. LANGSERVE ROUTER
# ============================================================

add_routes(

    app,

    formatted_agent_chain,

    path="/research-agent"

)


# ============================================================
# 17. HOME ENDPOINT
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
# 18. RUN APPLICATION
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
