#!/usr/bin/env python3
"""Small Google Sheets helper used by sheet_builder.py. Prints JSON on stdout.

Commands:
  status                                   show whether Google sign-in is set up (no network call)
  auth                                     sign in through the browser and store the token
  create --title T --tabs-json '["A","B"]' create a spreadsheet with the given tabs
  write --spreadsheet-id ID --tab NAME --rows-file rows.json
  batch-update --spreadsheet-id ID --requests-file requests.json

Sign-in files (first existing location wins):
  OAuth client file ("Desktop app" client downloaded from Google Cloud):
    $BRIEFPOST_GOOGLE_CLIENT_FILE
    ~/.config/briefpost-management/google-client.json
    ~/.claude/.credentials/sheets-client-secret.json
  Stored token (created by the "auth" command):
    $BRIEFPOST_GOOGLE_TOKEN_FILE
    ~/.config/briefpost-management/google-token.json
    ~/.claude/.credentials/sheets-oauth-token.json

Nothing in this file prints a token or a client secret.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

CONFIG_DIR = Path("~/.config/briefpost-management").expanduser()
SHARED_CREDENTIALS_DIR = Path("~/.claude/.credentials").expanduser()

MISSING_PACKAGES_HINT = (
    "Google packages are not installed. Run: "
    "python3 -m pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
)


class HelperError(Exception):
    """An error with a message that is safe to show to the user."""


def _first_existing(candidates: list[Optional[Path]]) -> Optional[Path]:
    for candidate in candidates:
        if candidate and candidate.is_file():
            return candidate
    return None


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def client_file() -> Optional[Path]:
    """Location of the OAuth client file, or None."""
    return _first_existing([
        _env_path("BRIEFPOST_GOOGLE_CLIENT_FILE"),
        CONFIG_DIR / "google-client.json",
        SHARED_CREDENTIALS_DIR / "sheets-client-secret.json",
    ])


def token_file() -> Optional[Path]:
    """Location of the stored token, or None."""
    return _first_existing([
        _env_path("BRIEFPOST_GOOGLE_TOKEN_FILE"),
        CONFIG_DIR / "google-token.json",
        SHARED_CREDENTIALS_DIR / "sheets-oauth-token.json",
    ])


def _token_save_path() -> Path:
    """Where a new token is stored: the env override, else this skill's own config folder."""
    return _env_path("BRIEFPOST_GOOGLE_TOKEN_FILE") or (CONFIG_DIR / "google-token.json")


def _save_token(credentials: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(credentials.to_json())
    os.chmod(path, 0o600)


def load_credentials(interactive: bool = False) -> Any:
    """Load (and refresh) the stored Google credentials.

    Args:
        interactive: If True and no valid token exists, open the browser sign-in.

    Raises:
        HelperError: If sign-in is not set up or the packages are missing.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
    except ImportError as exc:
        raise HelperError(MISSING_PACKAGES_HINT) from exc

    credentials = None
    stored = token_file()
    if stored:
        try:
            credentials = Credentials.from_authorized_user_file(str(stored), SCOPES)
        except Exception as exc:
            raise HelperError(f"Stored Google token at {stored} is not readable ({type(exc).__name__}). Run the 'auth' command again.") from exc

    if credentials and credentials.valid:
        return credentials

    if credentials and credentials.expired and credentials.refresh_token:
        try:
            credentials.refresh(Request())
            # A token file shared with another skill is never rewritten.
            if stored is None or SHARED_CREDENTIALS_DIR not in stored.parents:
                _save_token(credentials, stored or _token_save_path())
            return credentials
        except Exception as exc:
            if not interactive:
                raise HelperError(f"Google token could not be refreshed ({type(exc).__name__}). Run the 'auth' command again.") from exc

    if not interactive:
        raise HelperError(
            "Google sign-in is not set up. Put your OAuth client file at "
            f"{CONFIG_DIR / 'google-client.json'} and run: python3 {Path(__file__).name} auth"
        )

    client = client_file()
    if not client:
        raise HelperError(
            f"OAuth client file not found. Download a 'Desktop app' OAuth client from Google Cloud "
            f"and save it as {CONFIG_DIR / 'google-client.json'}"
        )
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError as exc:
        raise HelperError(MISSING_PACKAGES_HINT) from exc
    flow = InstalledAppFlow.from_client_secrets_file(str(client), SCOPES)
    credentials = flow.run_local_server(port=0)
    _save_token(credentials, _token_save_path())
    return credentials


def get_service(interactive: bool = False) -> Any:
    """Build the Sheets API client."""
    credentials = load_credentials(interactive=interactive)
    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise HelperError(MISSING_PACKAGES_HINT) from exc
    return build("sheets", "v4", credentials=credentials, cache_discovery=False)


# ---------------------------------------------------------------------------
# Commands. Each takes an optional ready-made service so it can be tested offline.
# ---------------------------------------------------------------------------

def cmd_status() -> dict[str, Any]:
    """Report which sign-in files exist. Makes no network call and reads no secret."""
    client = client_file()
    token = token_file()
    return {
        "client_file_found": bool(client),
        "client_file": str(client) if client else None,
        "token_found": bool(token),
        "token_file": str(token) if token else None,
        "ready": bool(token),
        "next_step": None if token else (
            f"Run: python3 {Path(__file__).resolve()} auth" if client else
            f"Save your OAuth client file as {CONFIG_DIR / 'google-client.json'}, then run the auth command"
        ),
    }


def cmd_auth() -> dict[str, Any]:
    load_credentials(interactive=True)
    return {"signed_in": True, "token_file": str(_token_save_path())}


def cmd_create(title: str, tabs: list[str], service: Any = None) -> dict[str, Any]:
    service = service or get_service()
    body = {
        "properties": {"title": title},
        "sheets": [
            {"properties": {"title": name, "gridProperties": {"frozenRowCount": 1}}}
            for name in tabs
        ],
    }
    created = service.spreadsheets().create(
        body=body, fields="spreadsheetId,spreadsheetUrl,sheets.properties",
    ).execute()
    return {
        "spreadsheet_id": created.get("spreadsheetId"),
        "url": created.get("spreadsheetUrl"),
        "sheets": [
            {
                "title": sheet.get("properties", {}).get("title"),
                "sheetId": sheet.get("properties", {}).get("sheetId"),
            }
            for sheet in created.get("sheets", [])
        ],
    }


def cmd_write(spreadsheet_id: str, tab: str, rows: list[list[Any]], service: Any = None) -> dict[str, Any]:
    service = service or get_service()
    escaped_tab = tab.replace("'", "''")
    response = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=f"'{escaped_tab}'!A1",
        valueInputOption="RAW",
        body={"values": rows},
    ).execute()
    return {"updated_cells": response.get("updatedCells", 0), "tab": tab}


def cmd_batch_update(spreadsheet_id: str, requests: list[dict[str, Any]], service: Any = None) -> dict[str, Any]:
    service = service or get_service()
    response = service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body={"requests": requests},
    ).execute()
    return {"replies": len(response.get("replies", []))}


def _read_json_file(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Google Sheets helper for the post summary")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("auth")
    p_create = sub.add_parser("create")
    p_create.add_argument("--title", required=True)
    p_create.add_argument("--tabs-json", required=True)
    p_write = sub.add_parser("write")
    p_write.add_argument("--spreadsheet-id", required=True)
    p_write.add_argument("--tab", required=True)
    p_write.add_argument("--rows-file", required=True)
    p_batch = sub.add_parser("batch-update")
    p_batch.add_argument("--spreadsheet-id", required=True)
    p_batch.add_argument("--requests-file", required=True)
    args = parser.parse_args(argv)

    try:
        if args.command == "status":
            result = cmd_status()
        elif args.command == "auth":
            result = cmd_auth()
        elif args.command == "create":
            result = cmd_create(args.title, json.loads(args.tabs_json))
        elif args.command == "write":
            result = cmd_write(args.spreadsheet_id, args.tab, _read_json_file(args.rows_file))
        else:
            result = cmd_batch_update(args.spreadsheet_id, _read_json_file(args.requests_file))
    except HelperError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # API errors can include request details, so only the type and a short message are shown
        print(f"ERROR: {type(exc).__name__}: {str(exc)[:300]}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
