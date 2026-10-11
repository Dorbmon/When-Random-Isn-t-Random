"""Verify the published inventory without executing collected source-code samples."""
import ast
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    total_rows = 0
    for record in manifest["files"]:
        path = root / record["path"]
        data = path.read_bytes()
        assert len(data) == record["bytes"], f"Size mismatch: {record['path']}"
        assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
        if path.suffix == ".json":
            json.loads(data)
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in data.splitlines() if line.strip()]
            assert len(rows) == record["rows"], record["path"]
            total_rows += len(rows)
        elif path.suffix == ".py" and "raw" not in path.relative_to(root).parts:
            ast.parse(data, filename=record["path"])
    print(f"Verified {len(manifest['files'])} files and {total_rows} JSONL rows.")


if __name__ == "__main__":
    main()
