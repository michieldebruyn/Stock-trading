"""Asks Claude for daily / weekly direction calls on a batch of tickers."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger(__name__)


class Call(BaseModel):
    symbol: str
    horizon: Literal["daily", "weekly"]
    direction: Literal["UP", "DOWN"]
    confidence: float = Field(description="Probability (0.50-0.95) that the direction is right")
    expected_move_pct: float = Field(description="Signed expected % change from base close to target close")
    rationale: str = Field(description="One short sentence")


class BatchResult(BaseModel):
    market_view: str
    predictions: list[Call]


OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "market_view": {"type": "string"},
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "horizon": {"type": "string", "enum": ["daily", "weekly"]},
                    "direction": {"type": "string", "enum": ["UP", "DOWN"]},
                    "confidence": {"type": "number"},
                    "expected_move_pct": {"type": "number"},
                    "rationale": {"type": "string"},
                },
                "required": ["symbol", "horizon", "direction", "confidence",
                             "expected_move_pct", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["market_view", "predictions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are the forecasting engine of a personal stock-prediction tracker. Every \
trading morning, before the markets open, you predict the price direction of stocks from the \
S&P 500, the largest European companies and the Belgian BEL20. Each evening your calls are \
scored against the actual closing prices, and your track record is shown back to you.

Definitions:
- daily: will the stock close today ABOVE (UP) or BELOW (DOWN) its last close (base close)?
- weekly: will the stock close on the last trading day of this week above or below the last \
close before this week started?
- confidence: your probability that the direction is right, between 0.50 and 0.95. Use it \
honestly: 0.50-0.55 means you see little edge. Your calibration is measured: when you say 0.70 \
you should be right about 70% of the time.
- expected_move_pct: the signed percentage change you expect from base close to target close \
(positive for UP, negative for DOWN), sized against the stock's typical volatility (atr14).
- rationale: one short, specific sentence naming the main drivers.

Input columns (all % unless noted): r1d/r5d/r20d/r60d = trailing returns; rsi14 = RSI (0-100); \
macd_h = MACD histogram as % of price; sma20/sma50/sma200 = distance of price from those moving \
averages; atr14 = average true range as % of price; vol_ratio = 5-day / 20-day average volume \
(x); from_52w_hi = distance from the 52-week high; last5_daily_% = the last five daily returns, \
oldest first; record = your last daily results on this stock, oldest first (+ right, - wrong); \
news = pre-market news research for this stock: a sentiment from -2 (clearly bad for today's \
price) to +2 (clearly good), then the catalyst. A blank news cell means nothing was found, not \
that nothing happened.

How to reason:
- Start from the market backdrop: overnight futures, Asian session, volatility, rates, FX. On \
most days the broad market drives most single-stock daily moves, so stocks in the same market \
should usually lean the same way unless there is a stock-specific reason.
- European stocks trade before the US opens; US futures and Asia are the main overnight signal \
for them. US stocks also react to the European morning.
- Then adjust per stock: momentum versus mean reversion, overbought/oversold readings, volume \
spikes, sector moves.
- News: a concrete catalyst announced since yesterday's close (earnings or guidance, M&A, an \
upgrade or downgrade, regulation) can outweigh the technicals, especially on earnings day. News \
that is days old is mostly priced in. Say in the rationale when news drove a call.
- Your playbook and the measured facts about your own habits are in the context below. They come \
from reviewing your real results: follow the rules, and fix the habits they show (lean, move \
size, overconfidence).
- Size expected_move_pct realistically: the context shows how far stocks actually move compared \
with what you expected.
- Short-horizon prediction is hard and close to a coin flip. Prefer modest confidence over \
bold calls. Use your track record to correct systematic bias (for example, too many UP calls) \
and overconfidence.

Return exactly one prediction for every requested (symbol, horizon) pair, using the symbol \
exactly as given. In market_view, write two or three sentences on today's overall outlook for \
the US and European markets."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    errors: list[str] = field(default_factory=list)


class ClaudePredictor:
    def __init__(self, model: str, effort: str = "medium", server_fallback: bool = True,
                 client: anthropic.Anthropic | None = None):
        self.model = model
        self.effort = effort
        self.server_fallback = server_fallback
        self.client = client or anthropic.Anthropic()
        self.usage = Usage()

    def predict_batch(self, run_context: str, rows: str, wanted: list[tuple[str, str]]) -> BatchResult:
        """`wanted` lists (symbol, horizon) pairs. Raises on API errors or unusable output."""
        user = user_message(rows, wanted)
        kwargs = dict(
            model=self.model,
            max_tokens=64000,
            # The system prompt and run context are identical for every batch in a run,
            # so they are cached after the first request.
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                {"type": "text", "text": run_context, "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": user}],
            thinking={"type": "adaptive"},
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        )
        if self.server_fallback:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        else:
            kwargs.update(betas=[])

        with self.client.beta.messages.stream(**kwargs) as stream:
            message = stream.get_final_message()

        self.usage.calls += 1
        u = message.usage
        self.usage.input_tokens += (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
            + (getattr(u, "cache_creation_input_tokens", 0) or 0)
        self.usage.output_tokens += u.output_tokens or 0

        if message.stop_reason == "refusal":
            raise RuntimeError(f"Claude declined the request: {getattr(message, 'stop_details', None)}")
        if message.stop_reason == "max_tokens":
            raise RuntimeError("Response hit max_tokens; lower claude.batch_size in config.yaml")

        text = next((b.text for b in message.content if b.type == "text"), None)
        if not text:
            raise RuntimeError(f"No text in response (stop_reason={message.stop_reason})")
        return parse_result(text, wanted)


def user_message(rows: str, wanted: list[tuple[str, str]]) -> str:
    pairs = "\n".join(f"{s} {h}" for s, h in wanted)
    return (
        f"Stocks:\n{rows}\n\n"
        f"Requested predictions ({len(wanted)}), one per line as 'SYMBOL horizon':\n{pairs}"
    )


def parse_result(text: str, wanted: list[tuple[str, str]]) -> BatchResult:
    """Validate a JSON answer and keep only the requested pairs, with confidence clamped to
    0.50-0.95 and the expected move's sign matching the direction."""
    try:
        result = BatchResult.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise RuntimeError(f"Unparseable response: {exc}") from exc

    wanted_set = set(wanted)
    clean, seen = [], set()
    for c in result.predictions:
        key = (c.symbol, c.horizon)
        if key not in wanted_set or key in seen:
            continue
        seen.add(key)
        c.confidence = min(max(c.confidence, 0.5), 0.95)
        # Keep the expected move consistent with the direction.
        c.expected_move_pct = abs(c.expected_move_pct) * (1 if c.direction == "UP" else -1)
        clean.append(c)
    result.predictions = clean
    return result
