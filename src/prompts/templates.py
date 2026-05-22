"""
Startup profile formatting for LLM input.

Converts raw Crunchbase rows into structured text prompts,
using anonymised text fields to avoid company name leakage.
"""
import json
import pandas as pd
from typing import Optional


def get_round_summary_from_row(row: pd.Series) -> Optional[str]:
    """
    Extract round-by-round funding summary from row's funding_round_details column.
    
    Expects 'funding_round_details' as list of dicts:
    [
        {'date': '2012-01-01', 'type': 'seed', 'amount': 500000},
        {'date': '2013-06-01', 'type': 'series_a', 'amount': 5000000},
    ]
    
    Returns formatted string like:
    "FUNDING DETAILS:
    - Seed (2012): $500,000
    - Series A (2013): $5,000,000"
    
    Returns None if no round details available.
    """
    round_details = row.get("funding_round_details")
    
    if not round_details or (isinstance(round_details, float) and pd.isna(round_details)):
        return None
    
    # Parse if stored as JSON string
    if isinstance(round_details, str):
        try:
            round_details = json.loads(round_details)
        except (json.JSONDecodeError, TypeError):
            return None
    
    if not isinstance(round_details, list) or len(round_details) == 0:
        return None
    
    lines = ["FUNDING DETAILS:"]
    for round_info in round_details:
        date_str = round_info.get("date", "")
        round_type = round_info.get("type", "unknown").replace("_", " ").title()
        amount = round_info.get("amount", 0)
        
        try:
            year = pd.Timestamp(date_str).year
            amount_str = f"${float(amount):,.0f}" if amount > 0 else "undisclosed"
            lines.append(f"  - {round_type} ({year}): {amount_str}")
        except Exception:
            pass
    
    return "\n".join(lines) if len(lines) > 1 else None


def format_startup_profile(row: pd.Series) -> str:
    """
    Format a single Crunchbase row as a startup profile string.

    Uses overview_anon (anonymised) as the primary textual self-description,
    supplemented by structured fundamentals. Field selection is guided by the
    SHAP feature importance ranking in Maarouf et al. (2025):
    TSD > startup age > founder education > sector > funding.

    Args:
        row: A pandas Series representing one company row.

    Returns:
        A formatted multi-line string suitable for injection into the user
        turn of a chat prompt.
    """
    lines = []

    # --- Textual self-description (most predictive feature per Maarouf) ---
    overview = row.get("overview_anon") 
    short_desc = row.get("short_description_anon")

    if overview and str(overview).strip():
        lines.append(f"DESCRIPTION:\n{str(overview).strip()}")
    elif short_desc and str(short_desc).strip():
        lines.append(f"DESCRIPTION:\n{str(short_desc).strip()}")
    else:
        lines.append("DESCRIPTION:\n[Not provided]")

    # --- Sector / category ---
    category = row.get("category_code")
    if pd.notna(category):
        lines.append(f"SECTOR: {category}")

    # --- Geography ---
    city = row.get("city")
    country = row.get("country_code")
    location_parts = [p for p in [city, country] if pd.notna(p)]
    if location_parts:
        lines.append(f"LOCATION: {', '.join(str(p) for p in location_parts)}")

    # --- Age / founding ---
    founded = row.get("founded_at")
    if pd.notna(founded):
        try:
            year = pd.Timestamp(founded).year
            lines.append(f"FOUNDED: {year}")
        except Exception:
            pass

    # --- Funding ---
    funding_total = row.get("funding_total_usd")
    funding_rounds = row.get("funding_rounds")
    funding_round_type = row.get("funding_round_type")
    if pd.notna(funding_total) and float(funding_total) > 0:
        lines.append(f"TOTAL FUNDING: ${float(funding_total):,.0f}")
    elif pd.notna(funding_rounds) and int(funding_rounds) == 0:
        lines.append("TOTAL FUNDING: None (pre-funding)")

    if pd.notna(funding_rounds):
        lines.append(f"FUNDING ROUNDS: {int(funding_rounds)}")

    # Round-by-round funding details (type, amount, date for each round)
    round_summary = get_round_summary_from_row(row)
    if round_summary:
        lines.append(round_summary)

    # Time since last funding round (proxy for funding recency / momentum)
    last_funding = row.get("last_funding_at")
    first_funding = row.get("first_funding_at")
    if pd.notna(last_funding) and pd.notna(first_funding):
        try:
            last_ts = pd.Timestamp(last_funding)
            first_ts = pd.Timestamp(first_funding)
            months = round((last_ts - first_ts).days / 30)
            lines.append(f"MONTHS BETWEEN FIRST AND LAST FUNDING: {months}")
        except Exception:
            pass

    # --- Network / relationships ---
    relationships = row.get("relationships")
    if pd.notna(relationships):
        lines.append(f"RELATIONSHIP COUNT: {int(relationships)}")

    # Investment activity (did the company itself invest in others?)
    inv_rounds = row.get("investment_rounds")
    if pd.notna(inv_rounds) and int(inv_rounds) > 0:
        lines.append(f"INVESTMENT ROUNDS MADE: {int(inv_rounds)}")

    # --- Team / founder signals ---
    team_size = row.get("team_size")
    if pd.notna(team_size) and int(team_size) > 0:
        lines.append(f"TEAM SIZE: {int(team_size)}")

    degree_count = row.get("person_with_degree_count")
    if pd.notna(degree_count):
        lines.append(f"TEAM MEMBERS WITH DEGREE: {int(degree_count)}")

    top_uni = row.get("any_top_university_person")
    if pd.notna(top_uni):
        lines.append(f"TOP UNIVERSITY ALUMNI ON TEAM: {'Yes' if top_uni else 'No'}")

    top_uni_count = row.get("top_university_person_count")
    if pd.notna(top_uni_count) and int(top_uni_count) > 0:
        lines.append(f"NUMBER OF TOP UNIVERSITY ALUMNI: {int(top_uni_count)}")

    return "\n".join(lines)
