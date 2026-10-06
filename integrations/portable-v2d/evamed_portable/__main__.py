"""Host CLI. Tool-call JSON is read from stdin; no credentials are accepted in JSON."""
import argparse
import json
from pathlib import Path
import sys
from .bundle import Bundle
from .integrity import Signer, strict_json
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["verify", "create", "call", "snapshot", "finalize"])
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--key")
    parser.add_argument("--run-root")
    parser.add_argument("--runs-root")
    parser.add_argument("--case-id")
    parser.add_argument("--bwrap")
    parser.add_argument("--runtime-root")
    args = parser.parse_args()
    bundle = Bundle(args.bundle)
    if args.operation == "verify":
        for case_id in bundle.cases:
            descriptor, *_ = bundle.case(case_id)
            from .integrity import contained_file, file_digest
            for item in descriptor["evidence"]:
                path = contained_file(bundle.root, item["path"])
                if path.stat().st_size != item["bytes"] or file_digest(path) != item["blake3"]:
                    raise ValueError("frozen_evidence_commitment")
        result = {"verified": True, "bundle_blake3": bundle.manifest["document_blake3"], "cases": len(bundle.cases)}
    else:
        if not all((args.key, args.bwrap, args.runtime_root)):
            parser.error("execution requires --key, --bwrap, --runtime-root")
        key = Path(args.key).resolve()
        if key.is_relative_to(bundle.root):
            parser.error("private signing key must be outside the distributable bundle")
        signer = Signer.create(key) if args.operation == "create" else Signer(key)
        common = dict(bundle=bundle, signer=signer, bwrap=args.bwrap, runtime_root=args.runtime_root)
        if args.operation == "create":
            if not args.case_id or not args.runs_root:
                parser.error("create requires --case-id and --runs-root")
            runtime = Runtime.create(case_id=args.case_id, runs_root=args.runs_root, **common)
            result = {"run_root": str(runtime.root), **runtime.public_snapshot()}
        else:
            if not args.run_root:
                parser.error("operation requires --run-root")
            runtime = Runtime(run_root=args.run_root, **common)
            if args.operation == "call":
                raw = sys.stdin.buffer.read(65537)
                if len(raw) > 65536:
                    raise ValueError("tool_request_too_large")
                request = strict_json(raw)
                result = runtime.call(request["name"], request["arguments"])
            elif args.operation == "snapshot":
                result = runtime.public_snapshot()
            else:
                result = runtime.finalize()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
