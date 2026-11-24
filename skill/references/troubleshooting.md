# Troubleshooting

Run the two checks first. They answer most questions.

```bash
BM=~/.claude/skills/briefpost-management
PY="$BM/.venv/bin/python"; [ -x "$PY" ] || PY=python3
"$PY" "$BM/scripts/selftest.py"                      # is the install healthy?
"$PY" "$BM/scripts/cli.py" ~/post-project --check    # is this project ready?
```

| What you see | Cause | Fix |
|--------------|-------|-----|
| `ERROR: no Gemini API key found` | No key in the shell or in the `.env` file | Put `GOOGLE_API_KEY=your_key` in `~/.config/briefpost-management/.env` |
| `[FAIL] ocr` with `API key not valid` | Key mistyped or deleted | Create a new key at https://aistudio.google.com/apikey |
| `[FAIL] ocr` with `404` or `model ... not found` | Google retired the model | Set newer model IDs under `ocr:` in `<project>/config.yaml` (list: https://ai.google.dev/gemini-api/docs/models) |
| `[FAIL] ocr` with `429` or `quota` | Free tier limit reached | Wait, then run again with `--resume`. Finished letters are not read again |
| `No photos (JPG, JPEG, PNG, HEIC) found` | Photos are in the wrong folder | Put them in `<project>/raw-images/unsorted/` |
| `is a HEIC photo and the macOS 'sips' tool was not found` | Not on a Mac | Export the photos as JPEG first |
| Several letters end up in one document | Photos taken too quickly one after another | Leave a minute between letters, or lower `grouping.doc_threshold_seconds` |
| Every case is `Open` although you paid | Bank CSV not recognised, or the payment reference is not in the bank text | Look for `CSV header detection failed` in the output and set `bank.columns` in `config.yaml` |
| `Google Sheet could not be created` | Google sign-in not set up | The local workbook is still written. Run `"$PY" "$BM/scripts/output/gsheets.py" status`, or set `output.destination: local` |
| `error: externally-managed-environment` when installing packages | Homebrew Python blocks global installs | Use the installer, it creates a private environment in `.venv` |
| Quality report shows FAIL lines | Individual checks did not pass | Read `reconciliation/qc-report.md`. The workbook is still written. Check the named cases against the letters |

## Starting again

Everything the pipeline writes is inside the project folder. To start from
scratch, delete these folders and run again. Your photos and bank files stay.

```bash
cd ~/post-project
rm -rf grouped classification ocr-pages ocr-letters letters reconciliation output qr-codes
```

## Where things are

| Path | Content |
|------|---------|
| `<project>/output/` | The workbook and CSV files |
| `<project>/reconciliation/qc-report.md` | Quality report |
| `<project>/letters/` | One JSON file per letter as it was read |
| `~/.config/briefpost-management/.env` | Your API key |
| `~/.config/briefpost-management/logs/ocr.log` | OCR log |
