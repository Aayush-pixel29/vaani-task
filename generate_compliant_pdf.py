# generate_compliant_pdf.py
import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

CSV_PATH = "results.csv"
PDF_OUT = "writeup.pdf"

# Paid-tier reference price stated explicitly for Table 2b cost computation ($0.10 / 1M tokens)
PRICE_PER_1K_TOKENS = 0.0001 


def compute_peak_concurrency(df_scen: pd.DataFrame) -> int:
    if df_scen.empty:
        return 0
    events = []
    for _, row in df_scen.iterrows():
        t0 = float(row["start_time"])
        dur_s = max(0.001, float(row["total_ms"]) / 1000.0)
        events.append((t0, 1))
        events.append((t0 + dur_s, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur, peak = 0, 0
    for _, delta in events:
        cur += delta
        if cur > peak:
            peak = cur
    return max(1, peak)


def pct_str(series: pd.Series, percentiles=(50, 95, 99)) -> str:
    valid = series[series > 0].dropna()
    if valid.empty:
        return "not measured"
    vals = [f"{np.percentile(valid, p):.1f}" for p in percentiles]
    return " / ".join(vals)


def build_tables_from_csv(csv_path: str):
    df = pd.read_csv(csv_path)

    # --- TABLE 2a & 2b ---
    scen_order = [
        "steady",
        "burst",
        "sustained_overload",
        "per_minute_limit_halved",
        "daily_budget_exhausted",
    ]
    # Also accept shorthand scenario names if logged that way
    scen_aliases = {
        "limit_halved": "per_minute_limit_halved",
        "daily_exhausted": "daily_budget_exhausted",
    }
    df["scenario_norm"] = df["scenario"].replace(scen_aliases)

    t2a_rows = [[
        "Scenario", "Real or\nstand-in", "Requests\nsent", "Duration\n(s)",
        "Offered rate\n(per min)", "First token p50 / p95 /\np99 (ms)",
        "Total time p50 / p95 /\np99 (ms)"
    ]]
    t2b_rows = [[
        "Scenario", "Succeeded\n%", "Rate-limited\n%", "Timed out or\ngiven up %",
        "Retries per\nrequest", "Peak queue\nlength", "Tokens per\nanswer",
        "Cost per 1,000\nanswers"
    ]]

    for sc in scen_order:
        sub = df[df["scenario_norm"] == sc]
        if sub.empty:
            continue
        n_req = len(sub)
        source = sub["source"].mode().iloc[0]
        t_min = sub["start_time"].min()
        t_max = (sub["start_time"] + sub["total_ms"] / 1000.0).max()
        dur_s = max(1.0, t_max - t_min)
        offered_rate = (n_req / dur_s) * 60.0

        ok_sub = sub[sub["status"] == "ok"]
        ttft_s = pct_str(ok_sub["first_token_ms"], (50, 95, 99))
        tot_s = pct_str(ok_sub["total_ms"] if not ok_sub.empty else sub["total_ms"], (50, 95, 99))

        t2a_rows.append([
            sc, source, str(n_req), f"{dur_s:.1f}", f"{offered_rate:.1f}", ttft_s, tot_s
        ])

        succ_pct = (sub["status"] == "ok").mean() * 100.0
        rl_pct = (sub["status"] == "rate_limited").mean() * 100.0
        to_pct = sub["status"].isin(["timeout", "error"]).mean() * 100.0
        retries_avg = sub["retries"].mean()
        peak_q = compute_peak_concurrency(sub)
        tot_tokens_series = sub["tokens_in"] + sub["tokens_out"]
        tok_per_ans = tot_tokens_series.mean()
        cost_1k = tok_per_ans * PRICE_PER_1K_TOKENS

        t2b_rows.append([
            sc, f"{succ_pct:.1f}%", f"{rl_pct:.1f}%", f"{to_pct:.1f}%",
            f"{retries_avg:.2f}", str(peak_q), f"{tok_per_ans:.0f}", f"${cost_1k:.2f}"
        ])

    # --- TABLE 3 ---
    t3_rows = [[
        "Setting (piece size, pieces\nfetched, re-rank yes/no)", "Questions\nasked",
        "Correct and\ncited %", "Wrong\n%", "Correctly\nrefused %",
        "Tokens per\nanswer", "First token p50 /\np95 (ms)", "Total time p50 /\np95 (ms)"
    ]]
    eval_df = df[df["scenario"] == "retrieval_eval"]
    for setting in ["256w_top3_norerank", "512w_top5_norerank", "512w_top5_rerank3"]:
        sub = eval_df[eval_df["retrieval_setting"] == setting]
        if sub.empty:
            continue
        n_q = len(sub)
        corr_pct = (sub["answer_grade"] == "correct").mean() * 100.0
        ref_corr_pct = (sub["answer_grade"] == "refused_correctly").mean() * 100.0
        # Wrong % includes both factual 'wrong' and 'refused_wrongly' so rows sum to 100.0%
        wrong_pct = sub["answer_grade"].isin(["wrong", "refused_wrongly"]).mean() * 100.0
        tok_ans = (sub["tokens_in"] + sub["tokens_out"]).mean()
        ttft_50_95 = pct_str(sub["first_token_ms"], (50, 95))
        tot_50_95 = pct_str(sub["total_ms"], (50, 95))

        label_map = {
            "256w_top3_norerank": "256w, top-3, re-rank: no",
            "512w_top5_norerank": "512w, top-5, re-rank: no",
            "512w_top5_rerank3":  "512w, top-5->3, re-rank: yes",
        }
        t3_rows.append([
            label_map.get(setting, setting), str(n_q),
            f"{corr_pct:.1f}%", f"{wrong_pct:.1f}%", f"{ref_corr_pct:.1f}%",
            f"{tok_ans:.0f}", ttft_50_95, tot_50_95
        ])

    return t2a_rows, t2b_rows, t3_rows


def make_rl_table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 6.3),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 6.2),
        ("LEADING", (0, 0), (-1, -1), 7.4),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 1.8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    return t


def generate_pdf():
    t2a_rows, t2b_rows, t3_rows = build_tables_from_csv(CSV_PATH)

    print("\n=== COMPUTED TABLE 2a FROM RESULTS.CSV ===")
    for row in t2a_rows:
        print(row)

    print("\n=== COMPUTED TABLE 2b FROM RESULTS.CSV ===")
    for row in t2b_rows:
        print(row)

    print("\n=== COMPUTED TABLE 3 FROM RESULTS.CSV ===")
    for row in t3_rows:
        print(row)

    doc = SimpleDocTemplate(
        PDF_OUT,
        pagesize=A4,
        leftMargin=0.38 * inch,
        rightMargin=0.38 * inch,
        topMargin=0.32 * inch,
        bottomMargin=0.32 * inch,
    )

    styles = getSampleStyleSheet()
    title_st = ParagraphStyle(
        "TitleSt", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=12.5, textColor=colors.HexColor("#0f172a"), spaceAfter=2
    )
    sub_st = ParagraphStyle(
        "SubSt", parent=styles["Normal"], fontName="Helvetica",
        fontSize=7.2, leading=8.8, textColor=colors.HexColor("#475569"), spaceAfter=4
    )
    h_st = ParagraphStyle(
        "HSt", parent=styles["Normal"], fontName="Helvetica-Bold",
        fontSize=8.0, leading=9.6, textColor=colors.HexColor("#0f172a"),
        spaceBefore=3.5, spaceAfter=1.2
    )
    body_st = ParagraphStyle(
        "BodySt", parent=styles["Normal"], fontName="Helvetica",
        fontSize=6.85, leading=8.35, textColor=colors.HexColor("#1e293b"), spaceAfter=2.0
    )

    story = []
    story.append(Paragraph("The Relay — Resilient Agentic RAG Benchmark & Systems Evaluation", title_st))
    story.append(Paragraph("Aayush Shelar (shelaraayush535@gmail.com) | Target Provider: Groq Free Tier (openai/gpt-oss-20b) & Calibrated FastAPI Stand-In", sub_st))

    # 1 What I built and why
    story.append(Paragraph("1 What I built and why", h_st))
    story.append(Paragraph(
        "<b>Architecture:</b> I built a 4-step asynchronous Python pipeline (<font name='Courier'>pipeline.py</font>) answering exclusively from arXiv "
        "1706.03762v7, 2410.00037v1, and 2507.07955v2: (1) <b>Query &amp; Paper Router</b> (deterministic keyword pre-router with a 45-token LLM fallback), "
        "(2) <b>Local BM25 Retrieval Tool</b> with a stateful section-number successor validator and glyph-density scrubber, (3) <b>Streaming Grounded Drafter</b> "
        "enforcing strict <font name='Courier'>[paper_id, p.X]</font> citations or an explicit <font name='Courier'>NOT_IN_PAPERS</font> refusal token, and "
        "(4) <b>Hybrid Critic &amp; Verification Loop</b> combining a 0-token regex page-citation gate with a targeted LLM grounding check and a 1-turn repair loop. "
        "<b>The One Decision That Shaped It Most:</b> Choosing <i>Dual-Stage Buffered Verification</i> over raw unverified streaming. Because the company never "
        "wants to send a wrong or embarrassing answer just to stay fast, tokens streamed from Groq are held in an internal buffer until Step 4 verifies that all "
        "citations exist in the retrieved pages and numbers are grounded. This deliberately increases user-visible median TTFT from approx. 420–814 ms (raw draft TTFT) "
        "to approx. 980–1,376 ms (verified release TTFT), trading approx. 560–950 ms of initial wait for 100% refusal accuracy on unanswerable queries.", body_st
    ))

    # 2 The real service and my stand-in
    story.append(Paragraph("2 The real service and my stand-in", h_st))
    story.append(Paragraph(
        "<b>Real Service Limits:</b> Probing Groq's free <font name='Courier'>on_demand</font> tier revealed that <font name='Courier'>llama-3.1-8b-instant</font> "
        "returned HTTP 404 (retired); active free-tier traffic runs on <font name='Courier'>openai/gpt-oss-20b</font>. Headers exposed "
        "<font name='Courier'>x-ratelimit-limit-tokens: 8000</font> (TPM, refilling via a continuous leaky bucket at 133.3 tokens/s) and "
        "<font name='Courier'>x-ratelimit-limit-requests: 1000</font>. By tracking <font name='Courier'>x-ratelimit-reset-requests</font> across sequential probes "
        "(7m12s = 432s after 5 requests, or 86.4s/request), I proved the 1,000 limit is <b>Requests Per Day (RPD)</b> (1,000 x 86.4s = 86,400s = 24h), while RPM "
        "and TPD are masked by the 8,000 TPM bottleneck. <b>Stand-In Calibration &amp; Differences:</b> <font name='Courier'>standin_server.py</font> is a FastAPI "
        "SSE server calibrated to Groq's empirical TTFT distribution, enforcing the 8,000 TPM (133.3 tok/s) and 1,000 RPD leaky buckets, emitting exact Groq 429 "
        "JSON payloads + <font name='Courier'>retry-after</font> headers, adding linear contention (+180 ms TTFT and +4 ms/token per concurrent request), 1.5% HTTP 500 "
        "faults, 2.5% tail-latency spikes (+1.5–2.5s), and streaming cached real Groq responses (<font name='Courier'>real_response_bank.json</font>). "
        "<b>Where it differs most:</b> It models contention linearly rather than simulating real LPU dynamic batching step-functions, lacks prefix KV-caching on "
        "repeated system prompts, and uses fixed reasoning-token overhead rather than input-dependent chain-of-thought length.", body_st
    ))

    t1_data = [
        ["Limit", "Value you found", "How you found it", "Confidence"],
        ["Requests per minute", "Not observed (masked by TPM)", "12-call burst hit 8,000 TPM after 6 calls before RPM triggered", "Low"],
        ["Tokens per minute", "8,000 TPM", "Header x-ratelimit-limit-tokens: 8000 & 429 body (includes reasoning tokens)", "High"],
        ["Requests per day", "1,000 RPD", "Header x-ratelimit-limit-requests: 1000 & reset timer +86.4s/call (1/1000 of 24h)", "High"],
        ["Tokens per day", "Not observed", "Not exposed in HTTP headers; daily token cap not reached during real runs", "Low"],
        ["Requests in flight at once", ">= 6 concurrent (no cap seen)", "6 of 12 parallel burst calls succeeded simultaneously; rest hit 429 TPM", "Medium"],
        ["What it returns when a limit is hit", "HTTP 429 JSON + retry-after", "Burst probe (Requested 1622, Remaining 0 -> code: rate_limit_exceeded)", "High"],
        ["Time until it recovers", "12.165s (retry-after: 13) for 1,622 tok", "Leaky-bucket refill at 133.3 tok/s (8,000/60s); verified by waiting retry-after", "High"],
    ]
    story.append(make_rl_table(t1_data, [1.35*inch, 1.55*inch, 3.65*inch, 0.90*inch]))
    story.append(Spacer(1, 2))

    # 3 Results
    story.append(Paragraph("3 Results", h_st))
    story.append(Paragraph(
        "<b>Real vs. Stand-In Runs:</b> All 17 calibration rows (<font name='Courier'>real_calibration</font>) and 49 of 54 retrieval evaluation rows in Table 3 "
        "(<font name='Courier'>retrieval_eval</font>) ran on the <b>real Groq service</b> (<font name='Courier'>source=real</font>, totaling 66 real calls; "
        "the final 5 unanswerable queries of <font name='Courier'>512w_top5_rerank3</font> ran on <font name='Courier'>stand-in</font>). "
        "The 5 load scenarios in Tables 2a/2b (112 requests total) ran against the calibrated <b>stand-in</b> (<font name='Courier'>source=stand-in</font>). "
        "One <font name='Courier'>results.csv</font> row equals one end-to-end question; <font name='Courier'>Peak queue length</font> is derived from interval overlap "
        "of <font name='Courier'>start_time</font> and <font name='Courier'>total_ms</font>. For Table 2a, latency percentiles are computed over completed answers "
        "(<font name='Courier'>status=ok</font>) where a first token was emitted, except <font name='Courier'>daily_budget_exhausted</font> where 100% fast-fail at "
        "5.3 / 10.5 / 15.5 ms. In Table 2b, <font name='Courier'>Tokens per answer</font> and <font name='Courier'>Cost per 1,000 answers</font> ($0.10/1M tokens) "
        "are averaged across all N offered requests, where 429-rejected and failed requests log 0 billed tokens; over <font name='Courier'>status=ok</font> rows only, "
        "mean tokens/answer are 2,745 in burst, 1,927 in sustained_overload, and 2,021 in per_minute_limit_halved (steady: 3,225 either way). "
        "<b>Two Numbers We Could Not Have Guessed Without Running It:</b> (1) <font name='Courier'>openai/gpt-oss-20b</font> streams 4–25 hidden "
        "reasoning tokens <i>before</i> the first visible answer token, inflating true user TTFT approx. 2.3x over raw first-chunk arrival while quietly burning the 8,000 TPM "
        "budget; (2) when Step 4 triggers a single Critic repair loop on <font name='Courier'>512w_top5_norerank</font> (e.g., <font name='Courier'>eval_512w_q05</font>), "
        "resending the multi-turn history balloons prompt consumption to <b>7,169 input tokens</b> (89.6% of the entire 8,000 TPM limit in one question!).", body_st
    ))

    story.append(make_rl_table(t2a_rows, [1.25*inch, 0.65*inch, 0.58*inch, 0.62*inch, 0.75*inch, 1.80*inch, 1.80*inch]))
    story.append(Spacer(1, 2))
    story.append(make_rl_table(t2b_rows, [1.25*inch, 0.72*inch, 0.78*inch, 0.85*inch, 0.75*inch, 0.75*inch, 0.95*inch, 1.40*inch]))

    # 4 Where the time goes
    story.append(Paragraph("4 Where the time goes", h_st))
    story.append(Paragraph(
        "<b>Single Slow Request Anatomy (<font name='Courier'>eval_512w_q05</font> on real Groq, <font name='Courier'>total_ms</font> = 3,548.3 ms, "
        "<font name='Courier'>first_token_ms</font> = 3,203.2 ms, <font name='Courier'>retries</font> = 1):</b> Splitting total latency across pipeline stages shows: "
        "<b>Waiting (Queue/Backoff) = 0.0 ms</b> (bucket had sufficient tokens at start); <b>Initial Working = 1,115.2 ms</b> (Step 1 Router + Step 2 BM25 tool took "
        "15.2 ms; Step 3 initial Drafter call took 1,100.0 ms); <b>Retrying / Critic Repair Loop = 2,088.0 ms</b> (Step 4a regex gate detected a missing bracketed "
        "page citation in the initial draft and triggered the repair loop, consuming an extra 3,560 prompt tokens and 2,088.0 ms of Groq inference); "
        "<b>Final Post-Verification Stream = 345.1 ms</b>. <b>Before vs. After First Token Proof:</b> 90.3% of total time (3,203.2 ms of 3,548.3 ms) occurred "
        "<i>before</i> the first user-visible token. We know this because step trace timestamps record <font name='Courier'>first_token_ms</font> only when Step 4 "
        "Critic approves the repaired buffer for release (<font name='Courier'>t_critic_ok</font> - <font name='Courier'>t_start</font> = 3,203.2 ms), proving the delay came from pre-release "
        "verification and repair rather than slow token streaming.", body_st
    ))

    # 5 When a limit is hit
    story.append(Paragraph("5 When a limit is hit", h_st))
    story.append(Paragraph(
        "<b>At the Per-Minute Limit (8,000 TPM):</b> Our pipeline tracks <font name='Courier'>x-ratelimit-remaining-tokens</font> from response headers. "
        "(1) <i>Who gets a weaker answer:</i> When remaining TPM drops below 3,800, adaptive retrieval automatically downgrades incoming queries from "
        "<font name='Courier'>512w_top5_norerank</font> (approx. 4,800 tokens/ans) to <font name='Courier'>256w_top3_norerank</font> (approx. 2,323 tokens/ans) and uses the "
        "0-token deterministic citation gate instead of the LLM Critic. This keeps requests succeeding under tight token budgets (<font name='Courier'>steady</font> "
        "and <font name='Courier'>per_minute_limit_halved</font>), at the cost of lower factual recall (61.5% vs. 76.9%). (2) <i>Who is delayed vs. dropped:</i> "
        "If a call still hits HTTP 429, in-flight requests enter full-jitter backoff (up to 2 retries, delaying tail completions to 39.2–44.3s in "
        "<font name='Courier'>burst</font> and <font name='Courier'>sustained_overload</font>). Requests exceeding <font name='Courier'>max_retries=2</font> are "
        "dropped cleanly as <font name='Courier'>rate_limited</font> (80.0% in burst, 70.0% in sustained_overload with 5.0% timed out) so they do not starve new arrivals. "
        "<b>At the Per-Day Limit (1,000 RPD):</b> Retrying a 24-hour window is futile; when <font name='Courier'>remaining-requests == 0</font> (or reset &gt; 60s), "
        "the circuit breaker trips immediately (5.3 ms p50 fast-fail, 10.5 ms p95, 0 retries) and falls back to local extractive BM25 snippets.", body_st
    ))

    # Retrieval: quality versus cost (Added after heading 5)
    story.append(Paragraph("Retrieval: quality versus cost", h_st))
    story.append(Paragraph(
        "Across 18 custom evaluation questions (13 answerable, 5 unanswerable traps) run on real Groq, <font name='Courier'>512w_top5_norerank</font> was selected as "
        "our primary default: it achieved the highest overall accuracy (<b>55.6% correct+cited [10/13 answerable = 76.9%]</b> and <b>27.8% correctly refused [5/5 = 100%]</b>, "
        "with only 16.7% wrong) at 4,990 total tokens/answer. <font name='Courier'>256w_top3_norerank</font> cut token cost by 49.4% (2,525 tokens/ans) and reduced "
        "p95 TTFT from 10,489.8 ms to 1,820.5 ms, but dropped answer recall to 8/13 (44.4% of 18) because small windows split hyperparameter tables. "
        "Surprisingly, <font name='Courier'>512w_top5_rerank3</font> performed worst on latency (p95 TTFT 22,942.1 ms due to extra repair loops) and reduced recall "
        "from 10/13 to 8/13 while slipping to 4/5 on refusals. When the TPM limit is close (&lt; 3,800 remaining tokens), the pipeline dynamically switches from "
        "<font name='Courier'>512w_top5_norerank</font> to <font name='Courier'>256w_top3_norerank</font> to cut token demand in half.", body_st
    ))

    story.append(make_rl_table(t3_rows, [1.65*inch, 0.62*inch, 0.72*inch, 0.58*inch, 0.72*inch, 0.72*inch, 1.22*inch, 1.22*inch]))

    # How this library will mislead a naive retriever (Added after heading 5)
    story.append(Paragraph("How this library will mislead a naive retriever", h_st))
    story.append(Paragraph(
        "<b>1. Cross-Paper Vocabulary Collision:</b> All three papers heavily share terms like <i>transformer, attention, tokens, layers,</i> and <i>latency</i>. "
        "On <b>Q01</b> (<i>\"What are N, d_model, and h in the base Transformer?\"</i>) or <b>Q12</b> (<i>\"What is the theoretical and practical latency of the spoken "
        "dialogue system?\"</i>), an unfiltered retriever pulls H-Net and Moshi architecture sections alongside Vaswani. Our Step 1 Paper Router prevents this via "
        "paper-ID metadata filtering. "
        "<b>2. Heuristic Section-Title Boosting Demoting Numeric Tables:</b> In <font name='Courier'>512w_top5_rerank3</font>, boosting chunks whose section title "
        "matches query words promoted prose overview sections (e.g., <i>Section 3.2.2 Multi-Head Attention</i>) above dense hyperparameter tables, dropping "
        "<b>Q01</b> (p.3 Table 3, N=6), <b>Q05</b> (Moshi p.3, Helium 7B / 2.1T tokens), and <b>Q12</b> (Moshi p.1, 200ms practical latency) during top-5-to-3 pruning. "
        "<b>3. PDF Layout &amp; Near-Miss Trap Contamination:</b> Two-column PDF extraction splits section numbers across lines (<font name='Courier'>3.1\\nEncoder</font>), "
        "turns table cells (<font name='Courier'>54.3 MPT</font>) into fake headings, and injects vertical diagram labels (<font name='Courier'>n d o f t o k e s</font> on "
        "Dynamic Chunking p.3, polluting <b>Q10</b>). On near-miss unanswerable traps like <b>Q15</b> (<i>\"How does RoPE compare against FlashAttention-2 in Attention Is "
        "All You Need?\"</i>), BM25 strongly matches <i>Attention</i> and <i>positional encoding</i>, misleading a naive drafter into guessing without our "
        "<font name='Courier'>NOT_IN_PAPERS</font> critic gate.", body_st
    ))

    # 6 Retries
    story.append(Paragraph("6 Retries", h_st))
    story.append(Paragraph(
        "<b>Policy:</b> We retry HTTP 429 and transient 5xx responses up to <font name='Courier'>max_retries = 2</font> using Full-Jitter Backoff anchored on the "
        "provider's header: <font name='Courier'>t_sleep = float(retry_after) + Uniform(0.5, 2.5)</font>, while never retrying daily budget exhaustion (RPD=0). "
        "<b>One Case Where Retrying Made Things Worse:</b> During our 12-request real burst probe (and replicated in the 25-request stand-in burst), Groq returned "
        "<font name='Courier'>retry-after: 13</font> simultaneously to every rate-limited caller. When honoring <font name='Courier'>retry-after</font> without jitter, "
        "all throttled requests woke up at the exact same millisecond 13.0s later. Because the leaky bucket had only refilled 1,733 tokens (13s * 133.3 tok/s)—barely "
        "enough for one call—the synchronized retry wave immediately collided into a secondary 429 thundering herd, burning retry budgets and inflating tail latency "
        "to 39.0s. Adding randomized jitter desynchronized wake-ups across the refill slope.", body_st
    ))

    # 7 What the person sees
    story.append(Paragraph("7 What the person sees", h_st))
    story.append(Paragraph(
        "At 3:00 PM when the daily budget is exhausted (<font name='Courier'>daily_budget_exhausted</font>), the user does <b>not</b> wait behind a doomed retry timer. "
        "Within 5.3 ms (p50; 10.5 ms p95), the assistant returns an explicit, honest fallback message containing deterministic local BM25 citations so the engineer is still "
        "unblocked: <i>\"[DAILY_AI_BUDGET_EXHAUSTED] Our daily language-model verification quota has been reached for today (resets at 00:00 UTC). Rather than guessing "
        "without verification, AI synthesis is paused. Here are the top exact passages retrieved locally from the arXiv library for your query: "
        "1) [moshi_2410.00037v1, p.2, Section 1 Introduction]: '...theoretical latency of 160ms and practical latency of 200ms...'\"</i>", body_st
    ))

    # 8 What I'm unsure about, what I left out, what I learned
    story.append(Paragraph("8 What I'm unsure about, what I left out, what I learned", h_st))
    story.append(Paragraph(
        "<b>Least Certain Assumption &amp; What Breaks:</b> Assuming Groq's 8,000 TPM bucket refills smoothly every millisecond rather than in discrete sub-second "
        "windows, and assuming our stand-in's linear contention (+180ms/request) matches LPU hardware batch scheduling. If Groq enforces micro-burst windowing or "
        "dynamic priority eviction, our 3,800-token degradation threshold could trigger too late under concurrent 512w repair loops. "
        "<b>Two Things Deliberately Left Out:</b> (1) <i>Dense Vector / Hybrid Embeddings:</i> Omitted to keep local retrieval latency at approx. 3 ms with zero external GPU/API "
        "dependencies across a 120-page corpus; (2) <i>Speculative Sentence-Level Streaming:</i> Omitted because streaming sentence 1 before sentence 2 passes the "
        "Critic risks mid-stream retraction if a later claim fails verification. "
        "<b>One Thing New to Me (In My Own Words):</b> How reasoning models distort streaming telemetry and token budgets. On <font name='Courier'>gpt-oss-20b</font>, "
        "the stream opens quickly (approx. 380 ms), but the first several chunks are empty role deltas and internal reasoning tokens (<font name='Courier'>delta.reasoning</font>) "
        "before any <font name='Courier'>delta.content</font> appears. Measuring socket-first-chunk instead of content-first-token understates user wait by more than half, "
        "and those unseen reasoning tokens silently drain the per-minute token bucket.", body_st
    ))

    # Your feedback on Vaani
    story.append(Paragraph("Your feedback on Vaani", h_st))
    story.append(Paragraph(
        "Visiting <font name='Courier'>https://www.vaani.dev/</font>, <b>what stood out</b> is the restraint and clarity of the positioning—focusing directly on "
        "natural conversational turn-taking and production voice reliability in Indian languages rather than generic AI buzzwords, backed by an immediately playable "
        "live audio demo. <b>What confused me</b> is the absence of technical transparency for systems engineers on the landing page: there are no published TTFT / "
        "voice-to-voice p50/p95 latency figures, interruption-handling benchmarks, or architectural docs explaining how tool calls avoid dead air. "
        "<b>One thing I would change:</b> Add a live <i>\"Latency &amp; Telemetry HUD\"</i> toggle right inside the web voice demo showing real-time VAD, ASR, LLM TTFT, "
        "and TTS audio-frame timestamps during a conversation—turning the demo itself into proof of engineering speed.", body_st
    ))

    # How I worked with the assistant
    story.append(Paragraph("How I worked with the assistant", h_st))
    story.append(Paragraph(
        "<b>Three Prompts That Mattered Most:</b> (1) Asking the assistant to translate the brief into concrete systems bottlenecks and cross-checking its output "
        "against a second model (Claude) to strip out unverified assumptions; (2) Directing the design of the 4-step pipeline with buffered verification and strict "
        "12-column <font name='Courier'>results.csv</font> logging; (3) Auditing the generated PDF structure against Page 2 and Page 3 of the brief to enforce exact "
        "heading and table compliance. "
        "<b>One Thing It Got Wrong That I Caught (and How):</b> The assistant proposed a regex section parser (<font name='Courier'>SEC_INLINE</font>) and a numeric "
        "grader using word-boundary patterns like <font name='Courier'>\\b8\\b</font> and <font name='Courier'>\\b6\\b</font>, plus initially suggested "
        "<font name='Courier'>llama-3.1-8b-instant</font>. I caught the model deprecation via live API probing (404 -> <font name='Courier'>openai/gpt-oss-20b</font>). "
        "I caught the parser bug by writing <font name='Courier'>test_user_parser.py</font> and inspecting its output across all 3 papers: it generated approx. 27 fake headings "
        "in Moshi and approx. 25 in Dynamic Chunking from numeric table cells (<font name='Courier'>54.3 MPT</font>, <font name='Courier'>0.2474 Watermarked</font>) and missed "
        "headings with parentheses (<font name='Courier'>2.2 Dynamic Chunking (DC)</font>). And I caught through unit tests that <font name='Courier'>\\b8\\b</font> "
        "falsely graded wrong answers as correct whenever they cited page 8 (<font name='Courier'>[p.8]</font>)! "
        "<b>One Decision Where I Overruled It:</b> The assistant proposed a heuristic bigram + section-title reranker (<font name='Courier'>512w_top5_rerank3</font>) "
        "expecting it to improve precision. Before wiring it in as default, I wrote an offline recall evaluator (<font name='Courier'>retrieval_recall.py</font>) across "
        "all 13 answerable questions and proved the reranker actually <i>dropped</i> recall from 13/13 to 10/13 by demoting hyperparameter tables—so I overruled it "
        "and made <font name='Courier'>512w_top5_norerank</font> the primary configuration. "
        "<b>What Is Distinctly Mine:</b> The successor-validated section state machine (yielding 23/39/35 clean headings), the citation-stripping grader with 8 self-tests, "
        "the <font name='Courier'>delta.reasoning</font> TTFT fix in the Groq probe, the 86.4s/request mathematical proof that <font name='Courier'>limit-requests: 1000</font> "
        "is RPD, and the empirical offline recall analysis.", body_st
    ))

    doc.build(story)
    print(f"Generated {PDF_OUT} successfully.")


if __name__ == "__main__":
    generate_pdf()

