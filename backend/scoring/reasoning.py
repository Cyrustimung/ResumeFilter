"""
Reasoning generator for CSV submission.

Writes like a recruiter's full assessment note  - natural, readable,
with specific evidence that justifies the ranking position.
No character limit  - as detailed as needed to explain the decision.
"""


def generate_reasoning(result: dict) -> str:
    """Generate a full, readable reasoning that explains the ranking decision."""
    f: dict = result["features"]
    p: dict = f["profile"]
    career: dict = f["career"]
    behavioral: dict = f["behavioral"]
    evidence = result.get("sub_scores", {}).get("evidence", {})
    if not isinstance(evidence, dict):
        evidence = {}

    role_audit = evidence.get("role_audit", {}) if isinstance(evidence.get("role_audit"), dict) else {}
    skill_audit = evidence.get("skill_audit", {}) if isinstance(evidence.get("skill_audit"), dict) else {}
    matched_kw = evidence.get("matched_keywords", {}) if isinstance(evidence.get("matched_keywords"), dict) else {}
    core_skills = skill_audit.get("core_skill_audit", []) if isinstance(skill_audit, dict) else []

    title = p["current_title"]
    years = p["years_exp"]
    title_cat = p.get("title_category", role_audit.get("title_category", "OTHER"))
    exp_verdict = role_audit.get("experience_verdict", "")
    career_ev = evidence.get("career_evidence", 0)

    # Gather proof points
    all_kw: list[str] = []
    for kws in matched_kw.values():
        if isinstance(kws, list):
            all_kw.extend(kws[:2])

    skill_proofs: list[str] = []
    for s in core_skills[:5]:
        if isinstance(s, dict) and s.get("verdict") in ("VERIFIED", "PARTIALLY_VERIFIED"):
            dur = s.get("duration_months", 0)
            name = s.get("name", "")
            if dur >= 24:
                years_dur = dur / 12
                skill_proofs.append(f"{name} ({years_dur:.1f}yr)")
            elif dur >= 12:
                skill_proofs.append(f"{name} ({dur}mo)")
            elif dur > 0:
                skill_proofs.append(f"{name}")

    response_rate = behavioral.get("response_rate", 0)
    notice_days = behavioral.get("notice_period_days", 0)
    github = behavioral.get("github_activity_score", -1)
    product_ratio = career.get("product_ratio", 0)
    avg_tenure = career.get("avg_tenure", 0)
    role_count = career.get("role_count", 0)

    parts: list[str] = []

    # ── WHO THEY ARE ──
    if title_cat == "AI_ML" and career_ev > 0.3:
        parts.append(f"{title} with {years:.1f} years whose career demonstrates production ML work  - career text references {', '.join(all_kw[:4])}.")
    elif title_cat == "AI_ML":
        parts.append(f"{title} with {years:.1f} years currently in AI/ML, though career descriptions lack strong production deployment evidence.")
    elif title_cat == "ADJACENT" and skill_proofs:
        parts.append(f"{title} with {years:.1f} years in technology, building depth in AI-relevant areas ({', '.join(skill_proofs[:2])}) but no direct ML production role yet.")
    elif title_cat == "ADJACENT":
        parts.append(f"{title} with {years:.1f} years in tech  - no ML production evidence found in career descriptions.")
    else:
        parts.append(f"{title} with {years:.1f} years  - non-technical profile, career background does not align with AI engineering requirements.")

    # ── WHAT THEY BRING (why they deserve this rank) ──
    brings: list[str] = []
    if title_cat == "AI_ML":
        brings.append("directly relevant current role in AI/ML")
    if career_ev > 0.3:
        brings.append("proven production ML work in career history")
    if skill_proofs:
        brings.append(f"verified skills: {', '.join(skill_proofs[:3])}")
    if exp_verdict == "IDEAL":
        brings.append(f"experience ({years:.1f}y) in the JD's ideal 6-8 year range")
    elif exp_verdict == "ACCEPTABLE":
        brings.append(f"experience ({years:.1f}y) within the JD's 5-9 year requirement")
    if product_ratio >= 0.7:
        brings.append(f"primarily product company background ({product_ratio:.0%})")
    if response_rate >= 0.7:
        brings.append(f"highly responsive to outreach ({response_rate:.0%} response rate)")
    if github >= 40:
        brings.append(f"active GitHub contributor (score {github:.0f}/100)")

    if brings:
        parts.append(f"Brings: {'; '.join(brings)}.")

    # ── EXPERIENCE FIT ──
    if exp_verdict == "TOO_JUNIOR":
        parts.append(f"At {years:.1f}y, falls below the JD's 5-year minimum for a senior role.")
    elif exp_verdict == "TOO_SENIOR":
        parts.append(f"At {years:.1f}y, may be overqualified for the seniority level described.")

    # ── CONCERNS (honest gaps) ──
    concerns: list[str] = []
    if career_ev < 0.1 and title_cat in ("AI_ML", "ADJACENT"):
        concerns.append("no ML production terms found in career text despite relevant skills")
    if notice_days > 90:
        concerns.append(f"long notice period ({notice_days}d, JD prefers sub-30)")
    if response_rate < 0.2 and response_rate > 0:
        concerns.append(f"low recruiter engagement ({response_rate:.0%} response rate)")
    if product_ratio == 0 and role_count > 1:
        concerns.append("entire career at consulting/services firms  - no product ownership")
    elif product_ratio < 0.3 and product_ratio > 0:
        concerns.append(f"only {product_ratio:.0%} at product companies")
    if 0 < avg_tenure < 16 and role_count >= 4:
        concerns.append(f"frequent job changes (avg {avg_tenure:.0f}mo across {role_count} roles)")

    if concerns:
        parts.append(f"Concerns: {'; '.join(concerns[:3])}.")

    return " ".join(parts)


