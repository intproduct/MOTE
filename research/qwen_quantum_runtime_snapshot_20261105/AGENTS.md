# Snapshot handling

- Payloads under `runtime`, `configs`, and `releases` are immutable copies of external experiment sources. Retain their original paths, formatting, and hashes.
- Do not invoke archived launchers, freeze/publish helpers, observers, or dashboard launchers just to validate this archive: some retain absolute paths into the live experiment and write state there.
- Use `verify_snapshot.py` for read-only integrity verification. The monitoring regression tests listed in `README.md` operate on local fixtures and mocks.
- Make future implementation changes in a new module or versioned snapshot. Preserve `SNAPSHOT.json` and the original payloads as provenance.
