---
name: briefpost-management
version: 6.0.0
description: >
  Turn a pile of paper post into a prioritised to-do list. Photograph your
  letters, and the skill reads them (Google Gemini), groups invoices with
  their reminders into cases, checks a bank CSV export to see what is already
  paid, ranks what is urgent, and writes a summary workbook (XLSX and CSV,
  optionally a Google Sheet). Use when the user says "sort my post",
  "process my letters", "which bills are still open", "check my post against
  my bank statement", or drops a folder of letter photos.
category: finance
tags: [ocr, post, letters, bills, reconciliation, bank-csv]
allowed-tools:
  - Bash
  - Read
  - Write
  - Edit
  - Glob
  - Grep
---

# Post sorter (briefpost-management)

Photos of letters in, a prioritised summary workbook out.

`photos -> group -> read (Gemini) -> letters -> cases -> bank check -> workbook + quality report`

Internal file and field names are German because the tool was first built for
German post (`vorgang` = case, `brief` = letter, `absender` = sender). The
workbook the user sees is in English. Letters in English and German are both
supported.

## Commands

```bash
BM=~/.claude/skills/briefpost-management
PY="$BM/.venv/bin/python"; [ -x "$PY" ] || PY=python3

# 0. Check the install (offline, about 2 seconds, no API key needed)
"$PY" "$BM/scripts/selftest.py"

# 1. Create a project folder
"$PY" "$BM/scripts/cli.py" ~/post-project --init

# 2. Check packages, API key and input files
"$PY" "$BM/scripts/cli.py" ~/post-project --check

# 3. Run everything
"$PY" "$BM/scripts/cli.py" ~/post-project

# Re-run from where it stopped
"$PY" "$BM/scripts/cli.py" ~/post-project --resume

# Run chosen phases only
"$PY" "$BM/scripts/cli.py" ~/post-project --phases match_vorgaenge,reconcile,cross_intelligence,output,qc
```

`cli.py` is the only entry point. Always call it with a project folder.

## What the user must provide

| Input | Where | Required |
|-------|-------|----------|
| Photos of letters (JPG, JPEG, PNG, HEIC) | `<project>/raw-images/unsorted/` | Yes |
| Gemini API key | `GOOGLE_API_KEY` in the shell or in `~/.config/briefpost-management/.env` | Yes, for reading the photos |
| Bank export as CSV | `<project>/bank-statements/` | No. Without it every case stays open |
| Personal settings | `<project>/config.yaml` | No. Defaults work for one household |

Never ask the user to paste the API key into the chat. Tell them where to put it.

Photograph tips to pass on: one page per photo, pages of the same letter one
after another within a few seconds, a short pause (a minute) before the next
letter. The grouping step uses the time between photos.

## Phases

| Phase | Reads | Writes | Needs API key |
|-------|-------|--------|---------------|
| `group` | `raw-images/unsorted/` | `grouped/doc-XXX/page-N.jpg`, `grouped/manifest.json` | No |
| `classify` | `grouped/` | `classification/classification-manifest.json` | Yes |
| `ocr` | `grouped/` + classification | `ocr-letters/*.json`, `ocr-pages/*.json` | Yes |
| `verify` | `ocr-letters/` | field checks added to `ocr-letters/` | Yes |
| `validate` | `ocr-letters/` | rule checks added | No |
| `postprocess` | `ocr-pages/` | cleaned `ocr-pages/` | No |
| `match_pages` | `ocr-pages/` | `letters/LTR-XXX.json` | No |
| `match_vorgaenge` | `letters/` | `reconciliation/vorgaenge.json` | No |
| `reconcile` | cases + `bank-statements/*.csv` | `reconciliation/vorgaenge-reconciled.json` | No |
| `cross_intelligence` | reconciled cases | `reconciliation/vorgaenge-cross-intel.json` | No |
| `output` | best available case file | `output/post-summary.xlsx`, four CSV files, `qr-codes/` | No |
| `qc` | everything (read only) | `reconciliation/qc-report.md` and `.json` | No |

## Result

`<project>/output/post-summary.xlsx` with four tabs: All letters, Open items
(most urgent first), Summary (totals, top creditors), Bank matching. The same
tabs are written as CSV files next to it.

How to read the labels:

| Workbook label | Internal value | Meaning |
|----------------|----------------|---------|
| Court (act now) | GERICHTLICH | Court order or enforcement. Has a legal deadline |
| Critical | KRITISCH | Final reminder, or a creditor marked critical in the settings |
| Soon | DRINGEND | Second reminder, or an open amount above the threshold |
| Can wait | NORMAL | Ordinary bill |
| Done | ERLEDIGT | Paid |
| Open / Paid / Part paid | OFFEN / BEZAHLT / TEILWEISE | Result of the bank check |

After a run, tell the user: how many letters and cases were found, what is
urgent (the first rows of Open items), the total still to pay, and where the
workbook is. Mention any failed checks from `reconciliation/qc-report.md`.

## Settings

Defaults live in `config/defaults.yaml`. Personal values go in
`<project>/config.yaml` (created by `--init`). Common ones:

| Setting | Use |
|---------|-----|
| `output.destination` | `local` (default), `google` or `both` |
| `output.currency` | Label next to totals. `EUR` also switches on SEPA payment QR codes |
| `persons.known_names` | Only if post for several people is mixed |
| `bank.columns`, `bank.date_format` | Only if the bank CSV is not recognised |
| `priority.critical_creditors` | Senders that must always rank at least Critical |
| `ocr.model` and the other model keys | Change if Google retires a model |

## Rules

1. Never delete, rename or edit the original photos in `raw-images/`.
2. Never round amounts. Keep them to the penny or cent.
3. Never change reference numbers. Copy them exactly.
4. A failed letter is flagged and skipped, the run continues.
5. Money decisions stay with the user. The workbook is a summary, not advice.
   Tell the user to check urgent items against the original letter.
6. Letters and bank exports are private. Do not paste their contents into
   other tools or services.

## Troubleshooting

See `README.md` in the giveaway folder, and `references/troubleshooting.md`.
