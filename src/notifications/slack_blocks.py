"""
Block Kit builders for interactive Slack approvals.

Pure functions — given already-formatted text and a pending-trade id, they emit
the `blocks` payload Slack needs to render Approve / Skip buttons. The webhook
already accepts `blocks`, so no bot token is required; button taps are delivered
to the /slack/interactive endpoint and the original message is updated in place
via the interaction's response_url.
"""

# action_id values the interactive endpoint dispatches on
ACTION_TAKE  = "trade_take"
ACTION_SKIP  = "trade_skip"
ACTION_CLOSE = "trade_close"
ACTION_HOLD  = "trade_hold"
ACTION_BUY   = "stock_buy"     # fallen-angel stock purchase


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _button(text: str, action_id: str, value: str, style: str | None = None) -> dict:
    b = {"type": "button", "text": {"type": "plain_text", "text": text}, "action_id": action_id, "value": value}
    if style:
        b["style"] = style
    return b


def entry_blocks(header: str, candidates: list[dict]) -> tuple[list, str]:
    """Build an approval message offering one Take/Skip pair per candidate.

    candidates: [{"tid": str, "text": str, "label": str}, ...]
    Returns (blocks, fallback_text).
    """
    blocks: list = [_section(header)]
    for i, c in enumerate(candidates, 1):
        blocks.append({"type": "divider"})
        blocks.append(_section(f"*{i}.* {c['text']}"))
        blocks.append({
            "type": "actions",
            "block_id": f"entry_{c['tid']}",
            "elements": [
                _button(f"✅ Take #{i}", ACTION_TAKE, c["tid"], "primary"),
                _button("✖ Skip", ACTION_SKIP, c["tid"], "danger"),
            ],
        })
    fallback = header + "\n" + "\n".join(f"{i}. {c['label']}" for i, c in enumerate(candidates, 1))
    return blocks, fallback


def exit_blocks(header: str, candidates: list[dict]) -> tuple[list, str]:
    """Build a close-approval message offering one Close/Hold pair per structure.

    candidates: [{"tid": str, "text": str, "label": str}, ...]
    """
    blocks: list = [_section(header)]
    for i, c in enumerate(candidates, 1):
        blocks.append({"type": "divider"})
        blocks.append(_section(f"*{i}.* {c['text']}"))
        close_label = c.get("close_label") or f"💰 Close #{i}"
        blocks.append({
            "type": "actions",
            "block_id": f"exit_{c['tid']}",
            "elements": [
                _button(close_label, ACTION_CLOSE, c["tid"], "primary"),
                _button("✋ Hold", ACTION_HOLD, c["tid"]),
            ],
        })
    fallback = header + "\n" + "\n".join(f"{i}. {c['label']}" for i, c in enumerate(candidates, 1))
    return blocks, fallback


def buy_blocks(header: str, candidates: list[dict]) -> tuple[list, str]:
    """Fallen-angel Buy/Skip approval. candidates: [{tid, text, label, ticker, low_52w}].
    The Buy button value is self-contained (fa_buy:SYM:LOW52W) so it survives a
    redeploy that wipes the ephemeral pending-store."""
    blocks: list = [_section(header)]
    for i, c in enumerate(candidates, 1):
        blocks.append({"type": "divider"})
        blocks.append(_section(f"*{i}.* {c['text']}"))
        buy_val = c.get("buy_value") or \
            f"fa_buy:{str(c.get('ticker','')).upper()}:{c.get('low_52w') or ''}"
        blocks.append({
            "type": "actions",
            "block_id": f"buy_{c['tid']}",
            "elements": [
                _button(f"🪂 Buy #{i}", ACTION_BUY, buy_val, "primary"),
                _button("✖ Skip", ACTION_SKIP, c["tid"], "danger"),
            ],
        })
    fallback = header + "\n" + "\n".join(f"{i}. {c['label']}" for i, c in enumerate(candidates, 1))
    return blocks, fallback


def resolved_blocks(text: str) -> list:
    """A plain single-section block set used to replace a message after the user
    has acted, so the buttons disappear and the outcome is recorded inline."""
    return [_section(text)]
