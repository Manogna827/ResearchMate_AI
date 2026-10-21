# ============================================================
# RESEARCHMATE AI
# Academic Research Assistant Agent
# ============================================================

import os
import uvicorn
import requests

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI

from langserve import add_routes


# ============================================================
# 1. GOOGLE API KEY
# ============================================================

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# 2. RESEARCH SEARCH TOOL
# ============================================================

@tool
def research_search(research_topic: str) -> str:
    """
    Search for academic research papers related to the
    research topic using the public OpenAlex API.
    """

    try:

        search_query = (
            research_topic
            + " CNN deep learning fruit disease plant disease "
              "image classification"
        )

        url = "https://api.openalex.org/works"

        params = {
            "search": search_query,
            "per-page": 6,
            "filter": "has_abstract:true",
        }

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("results", [])

        if not results:
            return (
                "No verified academic papers were found for this "
                "exact search query. Do not retry the search. "
                "Continue the research analysis using general "
                "domain knowledge and clearly state that paper "
                "retrieval was limited."
            )

        research_results = []

        for index, paper in enumerate(results, start=1):

            title = paper.get(
                "title",
                "Unknown Title"
            )

            year = paper.get(
                "publication_year",
                "Unknown Year"
            )

            doi = paper.get("doi")

            primary_location = (
                paper.get("primary_location")
                or {}
            )

            landing_page = primary_location.get(
                "landing_page_url"
            )

            if not landing_page:
                landing_page = doi

            if not landing_page:
                landing_page = "URL not available"

            source = (
                primary_location.get("source")
                or {}
            )

            source_name = source.get(
                "display_name",
                "Unknown Source"
            )

            abstract_inverted_index = paper.get(
                "abstract_inverted_index"
            )

            abstract = ""

            if abstract_inverted_index:

                words = []

                for word, positions in (
                    abstract_inverted_index.items()
                ):

                    for position in positions:

                        words.append(
                            (position, word)
                        )

                words.sort(
                    key=lambda item: item[0]
                )

                abstract = " ".join(
                    word
                    for _, word in words
                )

            if not abstract:
                abstract = (
                    "Abstract not available."
                )

            research_results.append(
                f"""
Research Paper {index}

Title:
{title}

Publication Year:
{year}

Source:
{source_name}

URL:
{landing_page}

Abstract:
{abstract}
"""
            )

        return "\n".join(
            research_results
        )

    except Exception as e:

        return (
            "Academic paper search was temporarily "
            "unavailable.\n\n"
            f"Search error: {str(e)}\n\n"
            "IMPORTANT: Do not retry the search. "
            "Continue the ResearchMate workflow using "
            "the available research knowledge and clearly "
            "state that external paper retrieval was limited."
        )


# ============================================================
# 3. RESEARCH ANALYSIS TOOL
# ============================================================

@tool
def research_analysis(
    research_topic: str,
    research_sources: str
) -> str:
    """
    Analyze the collected research information.
    """

    try:

        prompt = f"""
You are an academic research analyst.

Research Topic:
{research_topic}

Collected Research Sources:
{research_sources}

Analyze the available information.

Identify:

1. Research problem
2. Main research findings
3. Common methodologies
4. CNN architectures
5. Deep learning techniques
6. Image processing techniques
7. Datasets mentioned
8. Advantages
9. Limitations
10. Current research direction

Important rules:

- Use only information supported by the supplied research.
- Do not invent paper titles.
- Do not invent authors.
- Do not invent accuracy values.
- Do not invent datasets.
- Do not invent citations.
- If information is unavailable, clearly say so.
- General technical knowledge may be used for explanation,
  but clearly distinguish it from verified paper information.

Return a clear academic analysis.
"""

        model = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        response = model.invoke(prompt)

        return extract_model_text(response)

    except Exception as e:

        return (
            "Research analysis could not be completed.\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# 4. RESEARCH COMPARISON TOOL
# ============================================================

@tool
def research_comparison(
    research_topic: str,
    research_analysis_text: str
) -> str:
    """
    Compare the important research approaches and models.
    """

    try:

        prompt = f"""
You are a research comparison assistant.

Research Topic:
{research_topic}

Research Analysis:
{research_analysis_text}

Compare the important approaches used for this research topic.

Focus on:

1. CNN
2. Transfer Learning
3. ResNet
4. VGG
5. MobileNet
6. EfficientNet
7. Attention mechanisms
8. Data augmentation

For each relevant approach explain:

- Purpose
- Main advantage
- Main limitation
- Computational requirement
- Suitability for fruit disease detection

Do not invent numerical performance values.

If an approach was not explicitly present in the
provided research, explain it as general technical
background instead of presenting it as a verified
paper result.

Return a readable comparison.
"""

        model = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        response = model.invoke(prompt)

        return extract_model_text(response)

    except Exception as e:

        return (
            "Research comparison could not be completed.\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# 5. RESEARCH GAP TOOL
# ============================================================

@tool
def research_gap(
    research_topic: str,
    research_analysis_text: str,
    comparison_text: str
) -> str:
    """
    Identify possible research gaps and limitations.
    """

    try:

        prompt = f"""
You are an academic research-gap identification assistant.

Research Topic:
{research_topic}

Research Analysis:
{research_analysis_text}

Model Comparison:
{comparison_text}

Identify realistic research gaps.

Consider:

1. Dataset diversity
2. Real-world images
3. Different lighting conditions
4. Background variation
5. Small datasets
6. Class imbalance
7. Similar-looking diseases
8. Model size
9. Mobile deployment
10. Explainability
11. Generalization
12. Overfitting

Important:

Do not claim that a topic is completely unexplored.

Use phrases such as:

- "A possible research gap is..."
- "An area that can be further investigated is..."
- "Existing approaches may have limitations in..."

Do not invent research results.

Return a structured list of possible research gaps.
"""

        model = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        response = model.invoke(prompt)

        return extract_model_text(response)

    except Exception as e:

        return (
            "Research gap identification could not be completed.\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# 6. METHODOLOGY RECOMMENDATION TOOL
# ============================================================

@tool
def methodology_recommendation(
    research_topic: str,
    research_gap_text: str
) -> str:
    """
    Suggest a practical student-level research methodology.
    """

    try:

        prompt = f"""
You are a research methodology assistant.

Research Topic:
{research_topic}

Possible Research Gaps:
{research_gap_text}

Suggest a practical methodology for a student research project.

Include:

1. Problem definition
2. Dataset collection
3. Data preprocessing
4. Image resizing
5. Data augmentation
6. Train-validation-test split
7. Baseline CNN
8. Transfer learning model
9. Enhanced CNN idea
10. Model training
11. Evaluation metrics
12. Model comparison
13. Error analysis
14. Expected research contribution
15. Possible challenges

Suggested evaluation metrics can include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Do not invent experimental results.

Clearly distinguish expected methodology from
actual experimental results.

Return a practical step-by-step methodology.
"""

        model = ChatGoogleGenerativeAI(
            model="gemma-4-31b-it",
            temperature=0,
            google_api_key=GOOGLE_API_KEY
        )

        response = model.invoke(prompt)

        return extract_model_text(response)

    except Exception as e:

        return (
            "Methodology recommendation could not be completed.\n"
            f"Reason: {str(e)}"
        )


# ============================================================
# 7. TOOLS
# ============================================================

tools = [
    research_search,
    research_analysis,
    research_comparison,
    research_gap,
    methodology_recommendation
]


# ============================================================
# 8. MODEL
# ============================================================

model = ChatGoogleGenerativeAI(
    model="gemma-4-31b-it",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


# ============================================================
# 9. SYSTEM PROMPT
# ============================================================

system_prompt = """
You are ResearchMate AI.

You are an academic research assistant.

Your job is to analyze a research topic and produce
a useful research report for a student.

Follow this workflow:

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

IMPORTANT TOOL RULES:

1. Call research_search only ONCE.

2. After research_search returns, DO NOT call
   research_search again.

3. If research_search returns an error, timeout,
   empty result, or poor result, DO NOT retry it.

4. Continue with the remaining workflow.

5. Use research_analysis after research_search.

6. Use research_comparison after research_analysis.

7. Use research_gap after research_comparison.

8. Use methodology_recommendation after research_gap.

9. Do not create fake research papers.

10. Do not invent authors.

11. Do not invent datasets.

12. Do not invent accuracy values.

13. Do not invent experimental results.

14. Do not create fake citations.

15. If external research information is limited,
    clearly mention the limitation.

16. The final response must be readable text.

17. Do not expose internal tool calls or reasoning.

18. Do not stop after only one tool.

19. Complete the workflow whenever possible.

FINAL REPORT FORMAT:

RESEARCHMATE AI REPORT

Research Topic:

Research Question:

Research Goal:

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

The final report should be useful for a student
preparing a research paper or project.
"""


# ============================================================
# 10. CREATE AGENT
# ============================================================

agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt
)


# ============================================================
# 11. INPUT SCHEMA
# ============================================================

class AgentInput(BaseModel):

    research_topic: str = Field(
        ...,
        description="Research topic to investigate"
    )

    research_question: str = Field(
        ...,
        description="Main research question"
    )

    research_goal: str = Field(
        ...,
        description="Goal of the research"
    )


# ============================================================
# 12. FORMAT INPUT FOR AGENT
# ============================================================

def format_for_agent(input_data: AgentInput):

    return {
        "messages": [
            (
                "user",
                f"""

RESEARCHMATE AI INPUT

==================================================
RESEARCH TOPIC
==================================================

{input_data.research_topic}


==================================================
RESEARCH QUESTION
==================================================

{input_data.research_question}


==================================================
RESEARCH GOAL
==================================================

{input_data.research_goal}


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
            )
        ]
    }


# ============================================================
# 13. EXTRACT MODEL TEXT
# ============================================================

def extract_model_text(response):

    if response is None:
        return ""

    try:

        content = getattr(
            response,
            "content",
            None
        )

        if isinstance(content, str):

            return content.strip()

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):

                    text_parts.append(item)

                elif isinstance(item, dict):

                    if item.get("type") == "text":

                        text_parts.append(
                            item.get("text", "")
                        )

            return "\n".join(
                text_parts
            ).strip()

    except Exception:
        pass

    return str(response)


# ============================================================
# 14. EXTRACT FINAL AGENT RESPONSE
# ============================================================

def extract_text_response(response):

    if response is None:

        return (
            "ResearchMate AI could not complete the "
            "research workflow."
        )

    try:

        if isinstance(response, str):

            return response

        messages = response.get(
            "messages",
            []
        )

        for message in reversed(messages):

            content = getattr(
                message,
                "content",
                None
            )

            if content is None:

                if isinstance(message, dict):

                    content = message.get(
                        "content"
                    )

            if isinstance(content, str):

                if content.strip():

                    return content.strip()

            if isinstance(content, list):

                text_parts = []

                for item in content:

                    if isinstance(item, str):

                        text_parts.append(item)

                    elif isinstance(item, dict):

                        if item.get("type") == "text":

                            text_parts.append(
                                item.get(
                                    "text",
                                    ""
                                )
                            )

                text = "\n".join(
                    text_parts
                ).strip()

                if text:

                    return text

        return (
            "ResearchMate AI completed the workflow "
            "but no final report text was returned."
        )

    except Exception as e:

        return (
            "Unable to extract the final ResearchMate "
            f"report: {str(e)}"
        )


# ============================================================
# 15. LANGSERVE CHAIN
# ============================================================

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ============================================================
# 16. FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ResearchMate AI API",
    version="1.0.0",
    description="AI Agent for academic research assistance"
)


# ============================================================
# 17. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    formatted_agent_chain,
    path="/research-agent"
)


# ============================================================
# 18. HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "ResearchMate AI is running.",
        "agent_core": "gemma-4-31b-it",
        "endpoint": "/research-agent",
        "playground": "/research-agent/playground/"
    }


# ============================================================
# 19. RUN SERVER
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
