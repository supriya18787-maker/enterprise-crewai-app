import os
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM

import job_sources

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
  and similar non-profits (or companies that help non-profits), using free public job feeds (no API key needed)
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

def _keywords_from_text(roles: str):
    parts = [p.strip() for p in roles.replace("\n", ",").split(",")]
    kws = [p for p in parts if p]
    return kws or job_sources.DEFAULT_KEYWORDS


def curate_real_jobs(real_jobs: list, roles: str, location: str, max_results: int):
    """Pass REAL fetched postings to a CrewAI curator that formats them.

    The curator is explicitly grounded: it may only use the rows provided and
    must not invent jobs, links, or organisations.
    """
    job_curator = Agent(
        role="Job Listing Curator",
        goal="Organize REAL job postings into a clean, accurate, deduplicated listing for a candidate.",
        backstory=(
            "You turn raw job-feed data into a tidy, scannable job board. You ONLY use the "
            "postings handed to you and never invent jobs, links, or details."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )

    rows = "\n".join(
        f"- org: {j['organization']} | type: {j['type']} | title: {j['title']} | "
        f"location: {j.get('location') or 'N/A'} | url: {j.get('url') or 'N/A'}"
        for j in real_jobs[:max_results]
    )
    loc_note = f"\nCandidate location preference: {location}." if location.strip() else ""

    curate_task = Task(
        description=f"""
You are given REAL, already-fetched job postings (from official Greenhouse / Lever /
Workday feeds). Search intent: roles like "{roles}".{loc_note}

POSTINGS (use ONLY these — do not add or invent any):
{rows}

Produce a candidate-facing listing:
- One-line intro summarising the search and how many roles were found.
- A markdown table: Organization | Type | Job Title | Location | Apply Link.
  Render the Apply Link as a clickable markdown link using the provided url.
- Keep Gates Foundation / foundations first, then other non-profits/NGOs, then
  companies that help non-profits (the rows are already roughly in that order).
- Drop any obvious duplicates.
- If a location preference was given, you may note which roles best match it, but do
  NOT remove others.
- End with a one-line note that these are live feed results to confirm on the apply page.

Do not fabricate anything. Every row must come from the POSTINGS above.
""",
        expected_output="A clean markdown job listing table built only from the provided postings.",
        agent=job_curator
    )

    crew = Crew(agents=[job_curator], tasks=[curate_task], process=Process.sequential, verbose=False)
    return crew.kickoff()


def suggest_job_leads(roles: str, organizations: str, location: str, max_results: int):
    """Fallback when no real postings could be fetched (e.g. network blocked).

    Uses the model's knowledge to suggest organisations + official careers pages
    to check, clearly flagged as leads to verify (never confirmed live postings).
    """
    researcher = Agent(
        role="Nonprofit Tech Talent Researcher",
        goal=(
            "Suggest where to find Data Engineer / SQL / ETL roles at the Gates Foundation "
            "and similar non-profits, NGOs, foundations, and companies that help non-profits."
        ),
        backstory=(
            "You specialise in social-impact / non-profit technology hiring and know which "
            "organisations hire data engineers and where their official careers pages live."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False
    )
    loc = f" Prefer this location/region: {location}." if location.strip() else ""
    task = Task(
        description=f"""
No live job feed was reachable, so produce LEADS TO VERIFY (not confirmed postings).

Roles/keywords: {roles}
Organizations: {organizations}{loc}
List up to {max_results} organisations.

For each: Organization | Type (foundation/nonprofit/NGO/nonprofit-serving) |
Likely titles | Official careers-page URL.

Start with a bold disclaimer that these are leads to verify, not live postings.
Do NOT fabricate specific job-posting URLs — only official careers-page URLs.
""",
        expected_output="A markdown table of organisations and official careers pages to check, clearly marked as leads.",
        agent=researcher
    )
    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
    return crew.kickoff()


def run_nonprofit_job_search(roles, organizations, location, max_results, extra_sources=None):
    """On-demand task: list Data Engineer / SQL / ETL roles at the Gates
    Foundation and similar non-profits / nonprofit-serving companies.

    Strategy (no paid API needed):
      1. Pull REAL postings from free public ATS feeds (Greenhouse/Lever/Workday).
      2. Have a CrewAI curator format those real rows (grounded, no hallucination).
      3. If nothing could be fetched, fall back to model-suggested leads-to-verify.

    Returns (markdown_result, real_jobs, errors, mode) where mode is
    'live' (real feeds) or 'leads' (fallback).
    """
    keywords = _keywords_from_text(roles)
    real_jobs, errors = job_sources.fetch_jobs(keywords=keywords, extra_sources=extra_sources)

    if real_jobs:
        result = curate_real_jobs(real_jobs, roles, location, max_results)
        return result, real_jobs, errors, "live"

    result = suggest_job_leads(roles, organizations, location, max_results)
    return result, [], errors, "leads"


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
        "similar non-profits or companies that help non-profits — using FREE public job feeds "
        "(Greenhouse / Lever / Workday). No paid API key required."
    )
    st.info(
        "No API key needed. Real openings are pulled live from each employer's public careers feed. "
        "If outbound network is blocked (some sandboxes), it falls back to suggested leads to verify."
    )

    roles_input = st.text_input(
        "Roles / keywords (comma-separated)",
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
    max_results = st.slider("Max results", min_value=5, max_value=40, value=20, step=5)

    with st.expander("Add more organizations (optional, no key needed)"):
        st.caption(
            "One per line. Format: `greenhouse: <token> | Display Name` or `lever: <token> | Display Name`. "
            "The token is the company id in their careers URL "
            "(e.g. boards.greenhouse.io/**wikimedia**)."
        )
        extra_text = st.text_area("Extra sources", value="", height=100, label_visibility="collapsed")

    run_jobs_button = st.button("Run Job Search", type="primary")

    if run_jobs_button:
        if not roles_input.strip():
            st.warning("Please enter at least one role or keyword.")
        else:
            with st.spinner("Pulling live non-profit job feeds and curating with CrewAI..."):
                try:
                    extra_sources = job_sources.parse_extra_tokens(extra_text)
                    result, real_jobs, errors, mode = run_nonprofit_job_search(
                        roles=roles_input,
                        organizations=orgs_input,
                        location=location_input,
                        max_results=max_results,
                        extra_sources=extra_sources,
                    )

                    if mode == "live":
                        st.success(f"Found {len(real_jobs)} matching role(s) from live job feeds.")
                        st.subheader("Job Listings (live feeds)")
                        st.markdown(str(result))
                        with st.expander("Raw matched postings (source of truth)"):
                            st.dataframe(
                                [
                                    {
                                        "Organization": j["organization"],
                                        "Type": j["type"],
                                        "Title": j["title"],
                                        "Location": j.get("location", ""),
                                        "Apply": j.get("url", ""),
                                        "ATS": j.get("source_ats", ""),
                                    }
                                    for j in real_jobs[:max_results]
                                ],
                                use_container_width=True,
                            )
                    else:
                        st.warning(
                            "Could not reach the live job feeds from this environment, so these are "
                            "LEADS TO VERIFY (suggested organizations + official careers pages), not "
                            "confirmed live postings. Run the app where outbound HTTPS is allowed for live results."
                        )
                        st.subheader("Suggested leads to verify")
                        st.markdown(str(result))

                    if errors:
                        with st.expander(f"Sources skipped ({len(errors)})"):
                            for name, msg in errors:
                                st.text(f"{name}: {msg}")
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
