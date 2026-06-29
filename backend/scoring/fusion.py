"""
Score Fusion + Audit Report Generator.

USP: Every ranking decision comes with a full AUDIT TRAIL — not just a score,
but a traceable chain of evidence showing exactly WHY each decision was made.
A recruiter can verify every claim in seconds.
"""

from backend.config import WEIGHTS
from backend.pipeline.integrity_gate import compute_integrity_gate
from backend.pipeline.technical_fit import compute_evidence_score
from backend.pipeline.reliability import compute_reliability_score
from backend.pipeline.credibility import compute_credibility_score


def compute_final_score(features: dict) -> dict:
    """
    Compute final combined score for one candidate.
    Returns a full result with audit trail for complete transparency.
    """
    integrity: float = compute_integrity_gate(features)
    evidence, ev_audit = compute_evidence_score(features)
    reliability, rel_audit = compute_reliability_score(features)
    credibility, cred_audit = compute_credibility_score(features)

    raw: float = (
        WEIGHTS["technical_fit"] * evidence +
        WEIGHTS["reliability"] * reliability +
        WEIGHTS["credibility"] * credibility
    )
    final: float = integrity * raw

    # HARD CEILING: Candidates with ZERO ML evidence should NEVER outrank
    # candidates with genuine ML work.
    hard_failures = ev_audit.get("hard_failures", [])
    has_zero_ml = any("ZERO_ML_EVIDENCE" in f for f in hard_failures)
    has_domain_mismatch = any("DOMAIN_MISMATCH" in f for f in hard_failures)

    # RELEVANCE PENALTY: Technical fit drives everything.
    # Non-tech candidates should never outrank tech candidates with relevant skills.
    title_cat = features["profile"]["title_category"]
    skill_trust = ev_audit.get("skill_trust", 0)

    if has_zero_ml:
        # Zero ML evidence — strict ceiling based on title category
        if title_cat == "NON_TECH":
            # Non-tech + no ML = absolute floor. These should rank last.
            final = min(final, 0.08)
        elif title_cat == "ADJACENT":
            # Adjacent tech + no ML career evidence BUT might have AI skills
            # (this case is now less common because _check_hard_requirements
            #  gives credit for strong AI skills on ADJACENT titles)
            final = min(final, 0.14)
        else:
            final = min(final, 0.12)
    elif has_domain_mismatch:
        # Domain mismatch without zero_ml means some ML keywords exist
        # but title is non-tech — still heavily penalized
        final = min(final, 0.10)

    if evidence < 0.10 and not has_zero_ml:
        # Almost no evidence but passed the ML keyword check
        if title_cat == "NON_TECH":
            # Non-tech title with superficial ML mentions (e.g. "AI-strategy advisory")
            # These should NOT outrank adjacent-tech candidates
            final = min(final, 0.10)
        elif skill_trust > 0.3:
            # Adjacent/AI_ML title with genuine AI skills — transitioning engineer
            final = min(final, 0.25)
        else:
            final = min(final, 0.16)
    elif evidence < 0.20 and title_cat in ("NON_TECH", "OTHER"):
        # Non-tech title with very weak evidence
        if skill_trust > 0.3:
            final = min(final, 0.15)
        else:
            final = min(final, 0.10)
    elif evidence < 0.15 and title_cat == "ADJACENT":
        # Adjacent tech title but still limited ML work
        if skill_trust > 0.3:
            final = min(final, 0.25)
        else:
            final = min(final, 0.18)

    # Build the audit report
    # Micro-differentiation: add a tiny score variation (< 0.001) so candidates
    # at the same ceiling still have unique scores for deterministic ranking.
    # Uses reliability + credibility as natural tiebreaker signals.
    micro = (reliability * 0.0005 + credibility * 0.0003 + evidence * 0.0002)
    final = final + micro

    audit: dict = _build_full_audit(
        features, integrity, evidence, reliability, credibility,
        ev_audit, rel_audit, cred_audit
    )

    return {
        "candidate_id": features["candidate_id"],
        "final_score": round(final, 6),
        "integrity_gate": integrity,
        "evidence_score": round(evidence, 4),
        "reliability_score": round(reliability, 4),
        "credibility_score": round(credibility, 4),
        "sub_scores": {"evidence": ev_audit, "reliability": rel_audit, "credibility": cred_audit},
        "audit": audit,
        "features": features,
    }


def _build_full_audit(
    features: dict,
    integrity: float,
    evidence: float,
    reliability: float,
    credibility: float,
    ev_audit: dict,
    rel_audit: dict,
    cred_audit: dict,
) -> dict:
    """Build a complete human-readable audit report for the candidate."""
    p: dict = features["profile"]
    b: dict = features["behavioral"]
    career: dict = features["career"]

    # Score breakdown with contribution
    score_breakdown: dict = {
        "technical_fit_contribution": round(WEIGHTS["technical_fit"] * evidence, 4),
        "reliability_contribution": round(WEIGHTS["reliability"] * reliability, 4),
        "credibility_contribution": round(WEIGHTS["credibility"] * credibility, 4),
        "integrity_multiplier": integrity,
        "hard_multiplier": ev_audit.get("hard_multiplier", 1.0),
        "raw_score_before_integrity": round(
            WEIGHTS["technical_fit"] * evidence + WEIGHTS["reliability"] * reliability + WEIGHTS["credibility"] * credibility, 4
        ),
    }

    # Top reasons FOR this candidate
    positive_signals: list[str] = []
    if ev_audit.get("career_evidence", 0) > 0.3:
        keywords = ev_audit.get("matched_keywords", {})
        all_kw: list[str] = []
        for kws in keywords.values():
            all_kw.extend(kws[:2])
        positive_signals.append(f"Career evidence of relevant work (keywords: {', '.join(all_kw[:5])})")

    if ev_audit.get("skill_trust", 0) > 0.5:
        verified = [s["name"] for s in ev_audit.get("skill_audit", {}).get("core_skill_audit", []) if s["verdict"] == "VERIFIED"]
        if verified:
            positive_signals.append(f"Verified core skills: {', '.join(verified[:3])}")

    if rel_audit.get("response_rate", 0) >= 0.7:
        positive_signals.append(f"High recruiter response rate ({b['response_rate']:.0%})")

    if rel_audit.get("location", 0) >= 0.8:
        positive_signals.append(f"Location match: {p['location']}")

    if b["github_activity_score"] > 40:
        positive_signals.append(f"Active GitHub contributor (score: {b['github_activity_score']:.0f}/100)")

    if career["product_ratio"] > 0.7:
        positive_signals.append(f"Primarily product company experience ({career['product_ratio']:.0%})")

    role_audit = ev_audit.get("role_audit", {})
    if role_audit.get("title_category") == "AI_ML":
        positive_signals.append(f"Current role is AI/ML: {p['current_title']}")

    if role_audit.get("experience_score", 0) >= 0.7:
        positive_signals.append(f"Experience ({p['years_exp']:.1f}y) fits JD requirement of 5-9 years")

    # Top reasons AGAINST (concerns)
    negative_signals: list[str] = []
    if integrity < 1.0:
        negative_signals.append("⚠ Profile integrity issues detected (possible honeypot signals)")

    for hf in ev_audit.get("hard_failures", []):
        negative_signals.append(f"🚫 {hf}")

    for ap in ev_audit.get("anti_patterns", []):
        negative_signals.append(f"🚩 {ap['reason']}")

    if b["response_rate"] < 0.2:
        negative_signals.append(f"Low recruiter response rate ({b['response_rate']:.0%}) — may not be reachable")

    if b["last_active_days_ago"] > 180:
        negative_signals.append(f"Last active {b['last_active_days_ago']} days ago — may have left the platform")

    if b["notice_period_days"] > 60:
        negative_signals.append(f"Notice period: {b['notice_period_days']} days (JD prefers <30)")

    if p["title_category"] == "NON_TECH":
        negative_signals.append(f"Current role ({p['current_title']}) is non-technical — misaligned with Senior AI Engineer")

    if p["country"].lower() != "india" and not b["willing_to_relocate"]:
        negative_signals.append(f"Based in {p['country']}, not willing to relocate (JD requires India)")

    unverified = [s["name"] for s in ev_audit.get("skill_audit", {}).get("core_skill_audit", []) if s["verdict"] == "UNVERIFIED"]
    if unverified:
        negative_signals.append(f"Unverified skill claims: {', '.join(unverified[:3])} (listed but no career evidence)")

    return {
        "score_breakdown": score_breakdown,
        "positive_signals": positive_signals[:6],
        "negative_signals": negative_signals[:5],
        "evidence_trail": ev_audit.get("evidence_summary", []),
        "verdict": _compute_verdict(evidence, reliability, integrity),
    }


def _compute_verdict(evidence: float, reliability: float, integrity: float) -> str:
    """Human-readable one-line verdict."""
    if integrity == 0.0:
        return "REJECTED — Profile failed integrity checks (likely honeypot)"
    if integrity == 0.3:
        return "SUSPECT — Profile has integrity concerns, heavily penalized"
    if evidence > 0.6 and reliability > 0.6:
        return "STRONG FIT — High evidence of relevant work + actively available"
    if evidence > 0.4 and reliability > 0.4:
        return "GOOD FIT — Solid evidence with acceptable availability"
    if evidence > 0.3:
        return "PARTIAL FIT — Some relevant evidence but gaps exist"
    if evidence > 0.1:
        return "WEAK FIT — Limited evidence of relevant experience"
    return "NOT A FIT — No meaningful evidence of relevant work"
