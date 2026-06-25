"""Free, key-free job sourcing for the Nonprofit Job Finder.

Many non-profits and mission-driven companies publish their open roles on
public applicant-tracking systems (Greenhouse, Lever) or Workday. Those feeds
are open JSON endpoints that require NO API key, so we can pull real, current
postings without any paid search service.

NOTE: This module makes outbound HTTPS calls to the employers' job feeds. It is
meant to run in your own environment (local machine / Streamlit Cloud). Some
sandboxes block outbound traffic; in that case the app falls back to the
model's knowledge and clearly says so.

The SOURCES list below is a curated, EDITABLE starting point. ATS tokens
occasionally change; an unreachable or renamed source simply returns nothing
and is skipped. Add your own organisations by appending to SOURCES or by
passing extra Greenhouse/Lever tokens through the UI.
"""

import json
import urllib.request
import urllib.error

# Default role keywords (case-insensitive substring match on the job title).
DEFAULT_KEYWORDS = [
    "data engineer",
    "data engineering",
    "sql",
    "etl",
    "elt",
    "analytics engineer",
    "data platform",
    "data warehouse",
    "data pipeline",
]

# Curated mission-driven employers with public, key-free job feeds.
#   type: foundation | nonprofit | ngo | nonprofit-serving
# Tokens are best-effort; wrong/renamed ones fail gracefully and are skipped.
SOURCES = [
    # The organisation the request centres on (Workday).
    {"ats": "workday", "name": "Bill & Melinda Gates Foundation", "type": "foundation",
     "host": "gatesfoundation.wd1.myworkdayjobs.com", "tenant": "gatesfoundation", "site": "Gates"},

    # Non-profits / NGOs / foundations on Greenhouse or Lever.
    {"ats": "greenhouse", "name": "Wikimedia Foundation", "type": "nonprofit", "token": "wikimedia"},
    {"ats": "greenhouse", "name": "Mozilla", "type": "nonprofit", "token": "mozilla"},
    {"ats": "greenhouse", "name": "Code for America", "type": "nonprofit", "token": "codeforamerica"},
    {"ats": "greenhouse", "name": "The Trevor Project", "type": "nonprofit", "token": "thetrevorproject"},
    {"ats": "greenhouse", "name": "charity: water", "type": "nonprofit", "token": "charitywater"},
    {"ats": "greenhouse", "name": "Internet Archive", "type": "nonprofit", "token": "internetarchive"},
    {"ats": "lever", "name": "Khan Academy", "type": "nonprofit", "token": "khanacademy"},

    # Companies that primarily help / serve non-profits.
    {"ats": "greenhouse", "name": "DonorsChoose", "type": "nonprofit-serving", "token": "donorschoose"},
    {"ats": "greenhouse", "name": "Benevity", "type": "nonprofit-serving", "token": "benevity"},
    {"ats": "greenhouse", "name": "Classy", "type": "nonprofit-serving", "token": "classy"},
    {"ats": "greenhouse", "name": "Bonterra", "type": "nonprofit-serving", "token": "bonterra"},
    {"ats": "lever", "name": "Submittable", "type": "nonprofit-serving", "token": "submittable"},
]

# Rank ordering for display (Gates first, then non-profits, then helpers).
_TYPE_ORDER = {"foundation": 0, "nonprofit": 1, "ngo": 1, "nonprofit-serving": 2}

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NonprofitJobFinder/1.0)",
    "Accept": "application/json",
}


def _get_json(url, data=None, timeout=20):
    headers = dict(_HEADERS)
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    # urllib honours https_proxy / HTTPS_PROXY from the environment automatically.
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def fetch_greenhouse(token):
    url = f"https://boards-api.greenhouse.io/v1/boards/{token}/jobs?content=false"
    data = _get_json(url)
    out = []
    for j in data.get("jobs", []):
        out.append({
            "title": j.get("title", "").strip(),
            "location": (j.get("location") or {}).get("name", ""),
            "url": j.get("absolute_url", ""),
        })
    return out


def fetch_lever(token):
    url = f"https://api.lever.co/v0/postings/{token}?mode=json"
    data = _get_json(url)
    out = []
    for p in data:
        out.append({
            "title": p.get("text", "").strip(),
            "location": (p.get("categories") or {}).get("location", ""),
            "url": p.get("hostedUrl", ""),
        })
    return out


def fetch_workday(host, tenant, site, search_text=""):
    url = f"https://{host}/wday/cxs/{tenant}/{site}/jobs"
    body = json.dumps({
        "appliedFacets": {},
        "limit": 20,
        "offset": 0,
        "searchText": search_text,
    }).encode("utf-8")
    data = _get_json(url, data=body)
    out = []
    for j in data.get("jobPostings", []):
        path = j.get("externalPath", "")
        out.append({
            "title": j.get("title", "").strip(),
            "location": j.get("locationsText", ""),
            "url": f"https://{host}{path}" if path else "",
        })
    return out


def _matches(title, keywords):
    t = (title or "").lower()
    return any(k.lower() in t for k in keywords)


def fetch_jobs(keywords=None, extra_sources=None):
    """Fetch and keyword-filter real postings across all sources.

    Returns (results, errors):
      results: list of dicts {organization, type, title, location, url, source_ats}
      errors:  list of (organization_name, error_message) for sources that failed
    """
    keywords = keywords or DEFAULT_KEYWORDS
    sources = SOURCES + list(extra_sources or [])
    results, errors = [], []

    for s in sources:
        try:
            ats = s["ats"]
            if ats == "greenhouse":
                jobs = fetch_greenhouse(s["token"])
            elif ats == "lever":
                jobs = fetch_lever(s["token"])
            elif ats == "workday":
                # Pull broadly, then filter locally so multi-keyword matching works.
                jobs = fetch_workday(s["host"], s["tenant"], s["site"], "")
            else:
                continue

            for j in jobs:
                if _matches(j["title"], keywords):
                    results.append({
                        "organization": s["name"],
                        "type": s["type"],
                        "title": j["title"],
                        "location": j.get("location", ""),
                        "url": j.get("url", ""),
                        "source_ats": ats,
                    })
        except Exception as e:  # network error, 404, JSON change, etc. — skip source.
            errors.append((s.get("name", s.get("token", "unknown")), str(e)))

    results.sort(key=lambda r: (_TYPE_ORDER.get(r["type"], 9), r["organization"], r["title"]))
    return results, errors


def parse_extra_tokens(text):
    """Parse user-supplied 'extra source' lines into source dicts.

    Accepted line formats (one per line):
        greenhouse: <token> [| Display Name]
        lever: <token> [| Display Name]
    """
    extras = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line or ":" not in line:
            continue
        ats, rest = line.split(":", 1)
        ats = ats.strip().lower()
        if ats not in ("greenhouse", "lever"):
            continue
        if "|" in rest:
            token, name = rest.split("|", 1)
            token, name = token.strip(), name.strip()
        else:
            token, name = rest.strip(), rest.strip()
        if token:
            extras.append({"ats": ats, "name": name, "type": "nonprofit-serving", "token": token})
    return extras
