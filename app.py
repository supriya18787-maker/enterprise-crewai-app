import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

st.set_page_config(
    page_title="Enterprise Requirement Analyzer",
    layout="wide"
)

st.title("Enterprise Requirement Analyzer")
st.markdown(
    """
This app uses CrewAI agents to analyze a business requirement in an enterprise-style workflow:
- Business Analyst Agent
- Risk and Dependency Agent
- Executive Summary Agent
"""
)

def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it in .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

llm = LLM(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2
)

st.sidebar.header("Input Details")

domain = st.sidebar.selectbox(
    "Business Domain",
    ["Banking", "Insurance", "Healthcare", "Retail", "Telecom", "Education", "General Enterprise"]
)

priority = st.sidebar.selectbox(
    "Priority",
    ["Low", "Medium", "High", "Critical"]
)

audience = st.sidebar.selectbox(
    "Target Audience",
    ["Product Manager", "Engineering Manager", "Leadership", "CTO", "Business Stakeholders"]
)

requirement_text = st.text_area(
    "Enter the business requirement",
    height=250,
    placeholder="Example: We need an AI-based assistant that reads incoming support tickets, classifies severity, identifies duplicate issues, and prepares a suggested response for the support team."
)

run_button = st.button("Run Enterprise Analysis")

def run_enterprise_analysis(requirement: str, domain: str, priority: str, audience: str):
    business_analyst = Agent(
        role="Senior Business Analyst",
        goal="Understand the business requirement clearly and convert it into a structured enterprise analysis.",
        backstory=(
            "You are an expert enterprise business analyst who works with product teams, "
            "engineering teams, and stakeholders. You convert raw requirements into clear, "
            "structured business insights."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    risk_reviewer = Agent(
        role="Risk and Dependency Reviewer",
        goal="Identify enterprise risks, assumptions, dependencies, compliance concerns, and delivery blockers.",
        backstory=(
            "You are an experienced enterprise delivery and governance specialist. "
            "You identify implementation risks, system dependencies, operational constraints, "
            "and possible compliance concerns."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    executive_summarizer = Agent(
        role="Executive Summary Specialist",
        goal="Prepare a crisp executive summary suitable for leadership review.",
        backstory=(
            "You are skilled at communicating with directors, CTOs, and senior leaders. "
            "You write clear, concise, business-focused summaries."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    analysis_task = Task(
        description=f"""
Analyze the following enterprise business requirement.

Business domain: {domain}
Priority: {priority}
Audience: {audience}

Requirement:
{requirement}

Create a structured analysis with:
1. Business objective
2. Problem statement
3. Key users/stakeholders
4. Functional expectations
5. Non-functional expectations
6. Suggested implementation approach
7. Success metrics
""",
        expected_output="A clear and well-structured enterprise business analysis in markdown format.",
        agent=business_analyst
    )

    risk_task = Task(
        description=f"""
Review the same requirement below and identify:

Requirement:
{requirement}

Provide:
1. Key risks
2. Assumptions
3. Dependencies
4. Compliance or governance concerns
5. Suggested mitigations
6. Delivery considerations

Make the response enterprise-oriented and practical.
""",
        expected_output="A practical enterprise risk and dependency review in markdown format.",
        agent=risk_reviewer
    )

    summary_task = Task(
        description=f"""
Using the insights from prior analysis, write a concise executive summary for a {audience}.

Context:
- Domain: {domain}
- Priority: {priority}
- Requirement: {requirement}

Create:
1. Executive summary
2. Recommended next steps
3. Leadership decision points

Keep it crisp, professional, and business-oriented.
""",
        expected_output="A crisp executive summary suitable for enterprise leadership review.",
        agent=executive_summarizer
    )

    crew = Crew(
        agents=[business_analyst, risk_reviewer, executive_summarizer],
        tasks=[analysis_task, risk_task, summary_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return result

if run_button:
    if not requirement_text.strip():
        st.warning("Please enter a business requirement first.")
    else:
        with st.spinner("Running enterprise analysis with CrewAI agents..."):
            try:
                result = run_enterprise_analysis(
                    requirement=requirement_text,
                    domain=domain,
                    priority=priority,
                    audience=audience
                )

                st.success("Analysis completed successfully.")
                st.subheader("Final Output")
                st.markdown(str(result))

            except Exception as e:
                st.error(f"Something went wrong: {e}")