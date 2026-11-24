# Post sorter: a Claude Code skill for paper post

Photograph your letters. Get back one workbook that tells you what is urgent,
what is already paid and what can wait.

## What it does

1. **Groups** your photos into documents, using the time between shots.
2. **Reads** every letter with Google Gemini: sender, type (bill, reminder,
   court letter), amount, due date, reference number, payment details.
3. **Matches** letters that belong together into cases, so a reminder sits
   with its original bill.
4. **Checks your bank export** (CSV) to see which bills are already paid.
5. **Ranks** what is left: court letters first, final reminders next, then
   the rest.
6. **Writes a workbook** (`post-summary.xlsx` plus CSV files) with four tabs:
   All letters, Open items, Summary, Bank matching. A Google Sheet is optional.
7. **Checks its own work** and writes a quality report.

Everything runs on your Mac. The only thing that leaves it is the photos of
your letters, which are sent to Google Gemini to be read (and the result
rows, if you choose the Google Sheet option).

It works with letters in English and German. It was first built for German
post, so some file and field names inside the project folder are German
(`vorgang` = case, `brief` = letter). The workbook is in English.

## What you need

| Thing | Needed for | Cost |
|-------|-----------|------|
| A Mac with Claude Code | Running the skill | Your Claude plan |
| Python 3.9 or newer | Running the scripts. Check with `python3 --version` | Free. If missing: `xcode-select --install` |
| A Google Gemini API key | Reading the photos | Google has offered a free tier with limits; heavier use is billed by Google. Check their current pricing before a big batch |
| A CSV export from your bank | Seeing what is already paid | Optional. Without it every bill is shown as open |
| A Google account with a Cloud project | Writing a Google Sheet | Optional. The local workbook needs no account |

## Setup (about 10 minutes)

### 1. Install

Open Terminal, go to this folder and run:

```bash
cd /path/to/briefpost-management
./install.sh
```

The installer copies the skill to `~/.claude/skills/briefpost-management`,
creates a private Python environment there, installs the packages (needs
internet) and runs an offline self-test. You should see
`SELF-TEST PASSED` and `Installed and working.`

It never overwrites your settings. Running it again later is safe.

### 2. Get a Gemini API key

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Create an API key and copy it.
3. Open the key file the installer created:

   ```bash
   open -e ~/.config/briefpost-management/.env
   ```

4. Change the last line so it reads (no `#`, no spaces, no quotes):

   ```
   GOOGLE_API_KEY=your_key_here
   ```

5. Save and close. Do not share this file and do not paste the key into a chat.

### 3. Google Sheet (optional, skip this at first)

Only do this if you want the result as a Google Sheet as well as a file.

1. Run `./install.sh --with-google-sheet` to add the Google packages.
2. In https://console.cloud.google.com create a project, then enable the
   **Google Sheets API** for it.
3. Under **APIs and services > OAuth consent screen**, set the app up as
   External and add your own Google address as a test user.
4. Under **Credentials**, create an **OAuth client ID** of type **Desktop app**
   and download the JSON file.
5. Save that file as `~/.config/briefpost-management/google-client.json`.
6. Sign in once (a browser window opens):

   ```bash
   BM=~/.claude/skills/briefpost-management
   "$BM/.venv/bin/python" "$BM/scripts/output/gsheets.py" auth
   ```

7. In your project's `config.yaml` set `output: {destination: both}`.

## First run

```bash
BM=~/.claude/skills/briefpost-management
PY="$BM/.venv/bin/python"

# 1. Create a project folder (any name, anywhere)
"$PY" "$BM/scripts/cli.py" ~/post-project --init
```

2. Photograph your letters with your phone and copy the photos into
   `~/post-project/raw-images/unsorted/`. JPG, PNG and iPhone HEIC all work.
   - One page per photo, flat, whole page visible, good light.
   - Pages of the same letter straight after each other.
   - Wait about a minute before starting the next letter. The skill uses the
     pause to tell letters apart.
3. Optional: download a CSV export from your online banking covering the last
   few months and put it in `~/post-project/bank-statements/`.
4. Check that everything is ready:

   ```bash
   "$PY" "$BM/scripts/cli.py" ~/post-project --check
   ```

5. Run it. Either tell Claude Code:

   > sort my post in ~/post-project

   or run it yourself:

   ```bash
   "$PY" "$BM/scripts/cli.py" ~/post-project
   ```

   Each letter takes a few model calls with a 4 second pause between them, so a big pile takes a while. You can stop and continue later with `--resume`.

6. Open the result:

   ```bash
   open ~/post-project/output/post-summary.xlsx
   ```

Start with a small batch (five to ten letters) to see how it behaves with
your post before you photograph a whole drawer.

### Try it without a key

The self-test runs the matching, bank check and workbook steps on seven
made-up letters. No key and no internet needed:

```bash
"$PY" "$BM/scripts/selftest.py" --keep
```

It prints the folder where it left the demo workbook.

## Reading the workbook

| Tab | What it shows |
|-----|---------------|
| Open items | Every unpaid case, most urgent first. Start here |
| All letters | One row per letter, with the case it belongs to |
| Summary | Totals still to pay and already paid, counts, biggest creditors |
| Bank matching | Which bank payment was matched to which case, and how |

| Priority | Meaning |
|----------|---------|
| Court (act now) | A court order or enforcement letter. These have legal deadlines |
| Critical | A final reminder, or a sender you marked as critical |
| Soon | A second reminder, or a large open amount |
| Can wait | An ordinary bill |
| Done | Paid according to your bank export |

The workbook is a summary made by software. It can misread a letter. Always
check urgent items and amounts against the paper before you pay or ignore
anything. It is not legal or financial advice.

## Settings

Your settings live in `config.yaml` inside the project folder (created by
`--init`, with examples). The ones people change most:

| Setting | What it does |
|---------|--------------|
| `output.currency` | Label next to totals (default `GBP`). `EUR` also switches on SEPA payment QR codes in `qr-codes/` |
| `output.destination` | `local`, `google` or `both` |
| `persons.known_names` | Use if post for several people is mixed, to separate it |
| `bank.columns` | Use if your bank's CSV is not recognised |
| `priority.critical_creditors` | Senders that should always rank at least Critical |
| `ocr.model` and friends | Gemini model names. Change them if Google retires a model |

## Troubleshooting

| What you see | Fix |
|--------------|-----|
| `ERROR: no Gemini API key found` | Put `GOOGLE_API_KEY=your_key` in `~/.config/briefpost-management/.env` (step 2 above) |
| `[FAIL] ocr` with `API key not valid` | The key is wrong. Create a new one and replace it in the `.env` file |
| `[FAIL] ocr` with `404` or `model not found` | Google retired the model. Put newer model names under `ocr:` in `config.yaml`. Current names: https://ai.google.dev/gemini-api/docs/models |
| `[FAIL] ocr` with `429` or `quota` | You hit the free limit. Wait, then run again with `--resume`. Letters already read are not read again |
| `No photos (JPG, JPEG, PNG, HEIC) found` | The photos are not in `raw-images/unsorted/` inside the project folder |
| Several letters merged into one | Leave a longer pause between letters, or lower `grouping.doc_threshold_seconds` in `config.yaml` |
| Everything is `Open` although you paid | Your bank CSV was not understood, or the bank text does not contain the bill's reference. Look for `CSV header detection failed` in the output and set `bank.columns` |
| `Google Sheet could not be created` | The local workbook is still there. Run `gsheets.py status` (step 3 above), or set `output.destination: local` |
| `error: externally-managed-environment` | You installed packages by hand with Homebrew Python. Use `./install.sh`, it makes its own environment |
| `SELF-TEST FAILED` after install | Run `./install.sh` again with internet on. If it still fails, send the printed lines to whoever gave you the skill |

To start a project again from scratch, delete the generated folders. Your
photos and bank files are kept:

```bash
cd ~/post-project
rm -rf grouped classification ocr-pages ocr-letters letters reconciliation output qr-codes
```

## Privacy

- Your photos are sent to Google's Gemini API to be read. Google's API terms
  apply to them.
- Nothing is sent anywhere else. There is no tracking and no account with the
  author of this skill.
- Your API key stays in `~/.config/briefpost-management/.env` on your Mac.
- The project folder holds your letters as text. Keep it somewhere private.

## Uninstall

```bash
rm -rf ~/.claude/skills/briefpost-management ~/.config/briefpost-management
```

Your project folders are separate. Delete them yourself if you want them gone.
