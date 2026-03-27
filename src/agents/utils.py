def format_user_context(state) -> str:
    """Return a formatted string describing the authenticated user for LLM system prompts.

    Returns an empty string when no user context is available so callers can
    safely concatenate without extra conditional logic.
    """
    ctx: dict = state.get("user_context") or {}
    if not ctx:
        return ""

    lines = []
    if ctx.get("name"):
        lines.append(f"- Name: {ctx['name']}")
    if ctx.get("email"):
        lines.append(f"- Email: {ctx['email']}")
    if ctx.get("job_title"):
        lines.append(f"- Job title: {ctx['job_title']}")
    if ctx.get("department"):
        lines.append(f"- Department: {ctx['department']}")
    if ctx.get("company"):
        lines.append(f"- Company: {ctx['company']}")
    if ctx.get("country"):
        lines.append(f"- Country: {ctx['country']}")
    if ctx.get("manager"):
        lines.append(f"- Manager: {ctx['manager']}")

    if not lines:
        return ""

    return "\n\nUser context (the authenticated person you are talking with):\n" + "\n".join(lines)
