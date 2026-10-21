# ============================================================
# RESEARCHMATE AI - GEMINI ONLY
# ============================================================

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import uvicorn

from fastapi import FastAPI
from langserve import add_routes
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


# ============================================================
# 2. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY is not set. "
        "Please add GOOGLE_API_KEY in Render Environment Variables."
    )


# ============================================================
# 3. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)


# ============================================================
# 4. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    research_topic: str = Field(
        description="The research topic provided by the student"
    )

    research_question: str = Field(
        description="The research question provided by the student"
    )

    research_goal: str = Field(
        description="The research goal provided by the student"
    )


# ============================================================
# 5. TEXT EXTRACTION
# ============================================================

def extract_text(response):

    if response is None:
        return ""

    if isinstance(response, str):
        return response.strip()

    content = getattr(response, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, str):

                text_parts.append(item)

            elif isinstance(item, dict):

                item_type = item.get("type", "")

                if item_type in [
                    "thinking",
                    "reasoning"
                ]:
                    continue

                text = item.get("text")

                if text:
                    text_parts.append(str(text))

                output_text = item.get("output_text")

                if output_text:
                    text_parts.append(str(output_text))

        return "\n".join(text_parts).strip()

    return str(response).strip()


# ============================================================
# 6. SAFE GEMINI CALL
# ============================================================

def call_gemini(prompt):

    try:

        response = llm.invoke(prompt)

        result = extract_text(response)

        if result:

            return result

        return (
            "Gemini returned an empty response. "
            "Please try again."
        )

    except Exception as e:

        print("GEMINI ERROR:", str(e))

        return (
            "Gemini could not generate the research report "
            "at this moment. Please try again."
        )


# ============================================================
# 7. RESEARCHMATE AI WORKFLOW
# ============================================================

def researchmate_workflow(input_data):

    try:

        # ----------------------------------------------------
        # Read Input
        # ----------------------------------------------------

        if isinstance(input_data, AgentInput):

            research_topic = (
                input_data.research_topic.strip()
            )

            research_question = (
                input_data.research_question.strip()
            )

            research_goal = (
                input_data.research_goal.strip()
            )

        elif isinstance(input_data, dict):

            research_topic = str(
                input_data.get(
                    "research_topic",
                    ""
                )
            ).strip()

            research_question = str(
                input_data.get(
                    "research_question",
                    ""
                )
            ).strip()

            research_goal = str(
                input_data.get(
                    "research_goal",
                    ""
                )
            ).strip()

        else:

            return (
                "Invalid input. Please provide "
                "research_topic, research_question "
                "and research_goal."
            )


        # ----------------------------------------------------
        # Validate Input
        # ----------------------------------------------------

        if not research_topic:

            return "Please enter a research topic."

        if not research_question:

            return "Please enter a research question."

        if not research_goal:

            return "Please enter a research goal."


        # ----------------------------------------------------
        # ResearchMate AI Prompt
        # ----------------------------------------------------

        prompt = f"""
You are ResearchMate AI, an intelligent academic research
assistant.

Your job is to help a student understand a research topic,
analyze existing approaches, identify research gaps, compare
possible solutions, and recommend a practical research
methodology.

The student has provided:

RESEARCH TOPIC:
{research_topic}

RESEARCH QUESTION:
{research_question}

RESEARCH GOAL:
{research_goal}


============================================================
RESEARCHMATE AI WORKFLOW
============================================================

Follow these stages internally.

STAGE 1 - RESEARCH UNDERSTANDING

Understand the research topic.

Explain:

- What the research area is
- Why the topic is important
- Where it is commonly used
- What problem the research is trying to solve


STAGE 2 - EXISTING APPROACHES

Identify the important existing technical approaches related
to the topic.

For every important approach explain:

- Basic idea
- How it works
- Advantages
- Limitations
- Typical use cases


STAGE 3 - APPROACH COMPARISON

Compare the important approaches.

Consider:

- Accuracy
- Performance
- Complexity
- Computational requirements
- Data requirements
- Scalability
- Real-world applicability

Do not invent numerical results.


STAGE 4 - RESEARCH GAP IDENTIFICATION

Identify realistic research gaps.

Consider:

- Technical limitations
- Accuracy limitations
- Dataset limitations
- Generalization problems
- Computational limitations
- Real-world implementation problems
- Lack of robustness
- Possible improvements

Clearly distinguish between:

EXISTING KNOWLEDGE

and

PROPOSED RESEARCH IDEAS.


STAGE 5 - PROPOSED METHODOLOGY

Based on the research question, research goal and identified
gaps, recommend a practical methodology.

Include:

1. Problem definition
2. Data collection
3. Data preprocessing
4. Feature/model selection
5. Proposed technical approach
6. Training or implementation
7. Evaluation
8. Baseline comparison
9. Expected contribution


STAGE 6 - EXPECTED CONTRIBUTION

Explain what the proposed research could contribute.

Do not claim that the contribution has already been proven.


STAGE 7 - LIMITATIONS

Mention realistic limitations of the proposed research.


STAGE 8 - FINAL CONCLUSION

Give a concise academic conclusion connecting:

Research Topic
+
Research Question
+
Research Goal
+
Research Gap
+
Proposed Methodology


============================================================
IMPORTANT ACCURACY RULES
============================================================

You do NOT have live web search in this version.

Therefore:

- Do not claim that you searched the internet.
- Do not claim that you searched Google Scholar.
- Do not invent research papers.
- Do not invent authors.
- Do not invent DOI numbers.
- Do not invent URLs.
- Do not invent datasets.
- Do not invent experimental results.
- Do not invent accuracy percentages.
- Do not present proposed results as actual results.

Use your general academic knowledge.

If you are uncertain about a specific fact, describe it
generally instead of fabricating a citation.


============================================================
FINAL OUTPUT FORMAT
============================================================

Return ONLY the following report.

RESEARCHMATE AI REPORT

Research Topic:
{research_topic}

Research Question:
{research_question}

Research Goal:
{research_goal}


1. Research Overview

Explain the research area and importance.


2. Existing Approaches

Explain the major existing approaches.


3. Approach Comparison

Compare the approaches and explain their strengths
and weaknesses.


4. Research Gaps

Identify realistic research gaps.


5. Recommended Methodology

Give the proposed methodology step by step.


6. Expected Contribution

Explain the expected contribution.


7. Limitations

Explain realistic limitations.


8. Conclusion

Give a concise academic conclusion.


Use clear, simple academic English suitable for a
B.Tech/CSE student research project.

Do not include internal reasoning.
Do not include tool calls.
Do not include JSON.
Do not include Markdown code blocks.
"""


        # ----------------------------------------------------
        # Gemini Processing
        # ----------------------------------------------------

        print("\n========================================")
        print("RESEARCHMATE AI STARTED")
        print("========================================")

        print(
            "Research Topic:",
            research_topic
        )

        result = call_gemini(prompt)


        # ----------------------------------------------------
        # Final Output
        # ----------------------------------------------------

        print("\n========================================")
        print("RESEARCHMATE AI COMPLETED")
        print("========================================")

        return result


    except Exception as e:

        print(
            "WORKFLOW ERROR:",
            str(e)
        )

        return (
            "ResearchMate AI encountered an unexpected error. "
            "Please check your Google API key and try again."
        )


# ============================================================
# 8. LANGSERVE CHAIN
# ============================================================

researchmate_chain = (
    RunnableLambda(
        researchmate_workflow
    )
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 9. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ResearchMate AI",
    description=(
        "Gemini-powered Agentic AI Research Assistant "
        "for academic research analysis."
    ),
    version="1.0.0"
)


# ============================================================
# 10. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    researchmate_chain,
    path="/research-agent"
)


# ============================================================
# 11. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "ResearchMate AI is running.",
        "model": "gemini-2.5-flash",
        "service": "Gemini-only",
        "endpoint": "/research-agent",
        "playground": "/research-agent/playground/",
        "status": "ready"
    }


# ============================================================
# 12. HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ResearchMate AI",
        "model": "gemini-2.5-flash"
    }


# ============================================================
# 13. RUN APPLICATION
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
