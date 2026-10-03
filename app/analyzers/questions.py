"""Question bank used when the active analyzer has no LLM, or the LLM call fails."""

_BANK = [
    ("behavioral", "Tell me about a time you shipped something under a deadline you thought was impossible."),
    ("technical", "Walk me through how you would find why a p99 latency chart doubled after a deploy."),
    ("situational", "A teammate's change broke production on Friday. What do you do in the first hour?"),
    ("behavioral", "Tell me about a time you disagreed with a design and what you did about it."),
    ("technical", "How would you design a rate limiter for a public API? Talk about the numbers."),
    ("curveball", "What is a technical opinion you held strongly and then changed?"),
    ("behavioral", "Tell me about a time you had to explain a failure to someone senior."),
    ("situational", "You inherited a service with no metrics. What do you measure first, and why?"),
    ("technical", "Describe a bug that only showed up in production. How did you prove the fix?"),
    ("curveball", "What would you refuse to build, even if the team wanted it?"),
    ("behavioral", "Tell me about a time your answer in a meeting was too vague. What happened next?"),
    ("situational", "Your estimate was wrong by a week. How do you close that conversation?"),
]

_FOCUS = {
    "no_metrics": "Include the scale: requests, latency, people, or dollars.",
    "rambling": "Keep the story under two minutes.",
    "no_structure": "Use situation, action, and result.",
    "weak_closing": "End on the outcome.",
    "too_short": "Give enough detail that a stranger could retell it.",
    "off_topic": "Answer the question before you add context.",
    "unclear": "Use short sentences.",
}


def heuristic_questions(role: str, focus_tags: list[str], n: int = 5) -> list[str]:
    """Pick a varied set and bias the first questions toward her top tags."""
    role_bit = role.strip() or "this role"
    focused = []
    for tag in focus_tags:
        hint = _FOCUS.get(tag)
        if hint:
            stem = _BANK[len(focused) % len(_BANK)][1]
            focused.append(f"For a {role_bit} interview: {stem} {hint}")
    if not focused:
        focused.append(f"Why do you want a {role_bit} role, and what have you already shipped that looks like the job?")
    seen = set()
    questions: list[str] = []
    for question in focused + [item[1] for item in _BANK]:
        if question in seen:
            continue
        seen.add(question)
        questions.append(question)
        if len(questions) >= n:
            break
    return questions
