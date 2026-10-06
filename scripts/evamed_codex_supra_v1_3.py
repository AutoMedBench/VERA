#!/usr/bin/env python3
"""Inspect an explicit v1.3 route/mode without provider calls or credential reads."""
import argparse
import json

from eva_agent.codex_runtime.research_memory import ResearchContextPolicy
from eva_agent.codex_runtime.supra import SupraMode, SupraProfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "provider"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=[mode.value for mode in SupraMode], required=True)
    parser.add_argument("--protocol", choices=["codex_effort", "qwen_template", "chat_reasoning_effort"], required=True)
    parser.add_argument("--context-tokens", type=int, required=True)
    parser.add_argument("--compact-at-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, default=4096)
    parser.add_argument("--reserve-tokens", type=int, default=2048)
    parser.add_argument("--measured-compacted-input-tokens", type=int)
    parser.add_argument("--local-qwen-endpoint")
    args = parser.parse_args()
    try:
        profile = SupraProfile(model=args.model, provider=args.provider,
            mode=SupraMode(args.mode), protocol=args.protocol,
            context=ResearchContextPolicy(args.context_tokens, args.output_tokens,
                args.reserve_tokens, args.compact_at_tokens),
            local_qwen_endpoint=args.local_qwen_endpoint,
            measured_compacted_input_tokens=args.measured_compacted_input_tokens)
        print(json.dumps(profile.inspection(), sort_keys=True))
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
