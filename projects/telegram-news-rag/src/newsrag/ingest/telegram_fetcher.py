"""Download one day's files (newspaper/magazine PDFs) from a Telegram channel.

Usage:
    python -m newsrag.ingest.telegram_fetcher --dry-run          # list today's posts
    python -m newsrag.ingest.telegram_fetcher                    # download today's PDFs
    python -m newsrag.ingest.telegram_fetcher --date 2026-09-23  # a specific day
    python -m newsrag.ingest.telegram_fetcher --ids 60754 60757  # only these posts

Files land in data/raw/<date>/, with a manifest.jsonl recording the Telegram
metadata (message id, caption, timestamp) for each file. The next stages use
that metadata to tag chunks by date and publication.
"""

import argparse
import asyncio
import json
import re
from datetime import date, datetime, time, timedelta
from pathlib import Path

from telethon import TelegramClient
from telethon.tl.custom.message import Message

from newsrag.config import get_settings

DEFAULT_MIME_TYPES = ("application/pdf",)


def day_bounds(day: date) -> tuple[datetime, datetime]:
    """Start/end of `day` in the machine's local timezone (Telegram dates are UTC-aware)."""
    tz = datetime.now().astimezone().tzinfo
    start = datetime.combine(day, time.min, tzinfo=tz)
    return start, start + timedelta(days=1)


def safe_filename(name: str) -> str:
    return re.sub(r"[^\w.\- ]+", "_", name).strip() or "file"


def load_manifest(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["message_id"] for line in f if line.strip()}


async def iter_day_messages(client: TelegramClient, channel: str, day: date):
    """Yield the channel's messages posted on `day`, newest first."""
    start, end = day_bounds(day)
    # offset_date returns messages *older* than it, so start from the end of the day
    async for msg in client.iter_messages(channel, offset_date=end):
        if msg.date < start:
            break
        yield msg


def describe(msg: Message) -> str:
    local_time = msg.date.astimezone().strftime("%H:%M")
    if msg.file:
        size_mb = (msg.file.size or 0) / 1_000_000
        kind = f"{msg.file.mime_type or '?'}  {size_mb:6.1f} MB  {msg.file.name or '(no name)'}"
    else:
        kind = "text"
    caption = (msg.message or "").replace("\n", " ")[:60]
    return f"[{local_time}] #{msg.id:<7} {kind}  {caption}"


async def fetch_day(
    channel: str, day: date, mime_types: tuple[str, ...], dry_run: bool, ids: set[int] | None = None
) -> None:
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    out_dir = settings.raw_dir / day.isoformat()
    manifest_path = out_dir / "manifest.jsonl"
    already_have = load_manifest(manifest_path)

    # First run asks for your phone number and a login code in the terminal,
    # then reuses the saved session file.
    async with TelegramClient(
        str(settings.session_path), settings.telegram_api_id, settings.telegram_api_hash
    ) as client:
        # Private channels have no username, only a numeric id like -1001234567890
        entity = await client.get_entity(int(channel) if re.fullmatch(r"-?\d+", channel) else channel)
        print(f"Channel: {getattr(entity, 'title', channel)}  |  Day: {day}\n")

        matched = downloaded = 0
        async for msg in iter_day_messages(client, entity, day):
            if dry_run:
                print(describe(msg))
                continue
            if ids and msg.id not in ids:
                continue
            if not msg.file or msg.file.mime_type not in mime_types:
                continue
            matched += 1
            if msg.id in already_have:
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            name = safe_filename(msg.file.name or f"{msg.id}{msg.file.ext or ''}")
            target = out_dir / f"{msg.id}_{name}"
            print(f"Downloading {target.name} ({(msg.file.size or 0) / 1_000_000:.1f} MB)...")
            await client.download_media(msg, file=str(target))

            record = {
                "message_id": msg.id,
                "channel": channel,
                "posted_at": msg.date.isoformat(),
                "file": target.name,
                "original_name": msg.file.name,
                "mime_type": msg.file.mime_type,
                "size_bytes": msg.file.size,
                "caption": msg.message or "",
            }
            with manifest_path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            downloaded += 1

        if not dry_run:
            print(f"\n{matched} matching files, {downloaded} new, saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", help="Channel username or link (default: TELEGRAM_CHANNEL from .env)")
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(), help="YYYY-MM-DD (default: today)")
    parser.add_argument("--mime", nargs="+", default=list(DEFAULT_MIME_TYPES), help="MIME types to download")
    parser.add_argument("--ids", type=int, nargs="+", help="Only these message ids (the #numbers shown by --dry-run)")
    parser.add_argument("--dry-run", action="store_true", help="List the day's posts without downloading")
    args = parser.parse_args()

    channel = args.channel or get_settings().telegram_channel
    ids = set(args.ids) if args.ids else None
    asyncio.run(fetch_day(channel, args.date, tuple(args.mime), args.dry_run, ids))


if __name__ == "__main__":
    main()
