import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

# Optional live web-search tool. If crewai-tools / SERPER_API_KEY are not
# available the app still works using the model's own knowledge.
try:
    from crewai_tools import SerperDevTool
    _HAS_SERPER = True
except Exception:
    SerperDevTool = None
    _HAS_SERPER = False

st.set_page_config(
    page_title="Enterprise Requirement Analyzer",
    layout="wide"
)

st.title("Enterprise Requirement Analyzer")
st.markdown(
    """
This app uses CrewAI agents to run enterprise-style workflows:
- **Requirement Analyzer** — analyze a business requirement (Business Analyst, Risk, Executive Summary agents)
- **Nonprofit Job Finder** — an on-demand task that lists Data Engineer / SQL / ETL jobs at the Gates Foundation
  and similar non-profits (or companies that help non-profits)
"""
)

def get_secret(key: str, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o-mini")
SERPER_API_KEY = get_secret("SERPER_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing. Add it in .streamlit/secrets.toml or Streamlit Cloud secrets.")
    st.stop()

llm = LLM(
    model=OPENAI_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.2
)


def build_search_tool():
    """Return a configured SerperDevTool if possible, else None.

    Live web search needs both the crewai-tools package and a SERPER_API_KEY.
    When either is missing we return None and the crew falls back to the
    model's own knowledge (clearly flagged in the output).
    """
    if not (_HAS_SERPER and SERPER_API_KEY):
        return None
    # SerperDevTool reads SERPER_API_KEY from the environment.
    os.environ.setdefault("SERPER_API_KEY", str(SERPER_API_KEY))
    try:
        return SerperDevTool()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Feature 1: Enterprise Requirement Analyzer
# ---------------------------------------------------------------------------

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

    return crew.kickoff()


# ---------------------------------------------------------------------------
# Feature 2: Nonprofit Job Finder (on-demand scheduled task)
# ---------------------------------------------------------------------------

def run_nonprofit_job_search(roles: str, organizations: str, location: str, max_results: int):
    """On-demand task: find Data Engineer / SQL / ETL style jobs at the Gates
    Foundation and similar non-profits or companies that serve non-profits.

    Uses live web search when a SerperDevTool is available, otherwise falls
    back to the model's knowledge with an explicit disclaimer.
    """
    search_tool = build_search_tool()
    using_live_search = search_tool is not None
    tools = [search_tool] if using_live_search else []

    job_researcher = Agent(
        role="Nonprofit Tech Talent Researcher",
        goal=(
            "Find currently advertised data/engineering jobs (Data Engineer, SQL, ETL "
            "and closely related data roles) at the Gates Foundation and at similar "
            "non-profit organizations, foundations, NGOs, and companies that primarily "
            "serve or help non-profits."
        ),
        backstory=(
            "You are a specialist sourcer focused on the social-impact and non-profit "
            "technology sector. You know the major foundations (e.g. the Bill & Melinda "
            "Gates Foundation), NGOs, and mission-driven tech companies, and you are good "
            "at locating open data-engineering roles and their official application links."
        ),
        llm=llm,
        tools=tools,
        verbose=False,
        allow_delegation=False
    )

    job_curator = Agent(
        role="Job Listing Curator",
        goal="Organize raw job findings into a clean, accurate, deduplicated listing for a candidate.",
        backstory=(
            "You turn messy search results into a tidy, scannable job board. You never "
            "invent jobs, links, or details — if something is uncertain you flag it clearly."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    location_clause = f"Focus on this location/region: {location}." if location.strip() else "Location: any (remote-friendly preferred)."

    if using_live_search:
        sourcing_guidance = (
            "Use the web search tool to find CURRENTLY OPEN roles. Search official "
            "careers pages and reputable job boards. Capture the org name, exact job "
            "title, location/remote, a one-line summary, and the direct application URL. "
            "Prefer primary sources (the employer's own careers site)."
        )
    else:
        sourcing_guidance = (
            "NOTE: No live web search is configured (set SERPER_API_KEY to enable it). "
            "Work from your own knowledge. List the organizations most likely to hire for "
            "these roles and the typical titles they post, and provide their official "
            "careers-page URLs so the candidate can verify current openings themselves. "
            "Clearly mark that these are leads to verify, not confirmed live postings."
        )

    research_task = Task(
        description=f"""
On-demand task: find jobs matching the candidate's interest.

Target roles / keywords: {roles}
Target organizations: {organizations}
{location_clause}
Return up to {max_results} of the most relevant results.

{sourcing_guidance}

For every result gather, where available:
- Organization name (and whether it is a non-profit / foundation / NGO / company serving non-profits)
- Job title
- Location or remote status
- A one-sentence summary of the role
- Direct application / careers URL
- Source (where you found it)

Do NOT fabricate postings or URLs. If you cannot confirm a live posting, say so.
""",
        expected_output="A raw but organized set of relevant job leads with sources and links.",
        agent=job_researcher
    )

    curate_task = Task(
        description=f"""
Take the researcher's findings and produce the final candidate-facing listing.

Requirements:
- A short intro line stating the search ({roles} @ {organizations}{', ' + location if location.strip() else ''}).
- A markdown table with columns: Organization | Type | Job Title | Location | Apply Link | Source.
- Group or sort with the Gates Foundation first, then other non-profits / NGOs / foundations,
  then companies that help non-profits.
- Remove duplicates. Keep only roles genuinely related to Data Engineer / SQL / ETL / data work.
- Below the table, add a short "How to verify / next steps" note.
- If results are leads-to-verify rather than confirmed live postings, state that clearly at the top.

Keep it clean, accurate, and easy to scan.
""",
        expected_output="A clean markdown job listing with a table, sources, and verification notes.",
        agent=job_curator
    )

    crew = Crew(
        agents=[job_researcher, job_curator],
        tasks=[research_task, curate_task],
        process=Process.sequential,
        verbose=False
    )

    result = crew.kickoff()
    return result, using_live_search


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

tab_analyzer, tab_jobs = st.tabs(["Requirement Analyzer", "Nonprofit Job Finder"])

with tab_analyzer:
    st.sidebar.header("Requirement Analyzer Input")

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

with tab_jobs:
    st.subheader("Nonprofit Data/ETL Job Finder")
    st.caption(
        "On-demand task. Lists Data Engineer / SQL / ETL roles at the Gates Foundation and "
        "similar non-profits or companies that help non-profits."
    )

    if _HAS_SERPER and SERPER_API_KEY:
        st.info("Live web search is enabled (Serper). Results reflect current openings where found.")
    else:
        st.warning(
            "Live web search is not configured — results come from the model's knowledge and are "
            "leads to verify. Set SERPER_API_KEY (and install `crewai-tools`) for live listings."
        )

    roles_input = st.text_input(
        "Roles / keywords",
        value="Data Engineer, SQL, ETL"
    )
    orgs_input = st.text_input(
        "Organizations to target",
        value="Gates Foundation and similar non-profits, NGOs, foundations, or companies that help non-profits"
    )
    location_input = st.text_input(
        "Location / region (optional)",
        value="",
        placeholder="e.g. Seattle, USA, India, or leave blank for remote/any"
    )
    max_results = st.slider("Max results", min_value=5, max_value=30, value=15, step=5)

    run_jobs_button = st.button("Run Job Search", type="primary")

    if run_jobs_button:
        if not roles_input.strip():
            st.warning("Please enter at least one role or keyword.")
        else:
            with st.spinner("Searching for nonprofit data/ETL jobs with CrewAI agents..."):
                try:
                    result, used_live = run_nonprofit_job_search(
                        roles=roles_input,
                        organizations=orgs_input,
                        location=location_input,
                        max_results=max_results
                    )
                    st.success("Job search completed.")
                    if not used_live:
                        st.caption("Generated from model knowledge — verify each posting via the official links.")
                    st.subheader("Job Listings")
                    st.markdown(str(result))
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
