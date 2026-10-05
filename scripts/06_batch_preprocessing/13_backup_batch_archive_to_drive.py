"""Upload one batch ZIP archive to Google Drive and verify it.

Example:
    python 13_backup_batch_archive_to_drive.py \
        --batch-id BATCH_001 \
        --dry-run

    python 13_backup_batch_archive_to_drive.py \
        --batch-id BATCH_001 \
        --execute

This script:
- uploads only BATCH_xxx.zip
- uses the existing drive.file OAuth token
- verifies local MD5/SHA256
- verifies remote size/MD5
- never deletes local files
- never overwrites an existing mismatched Drive file
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path


ROOT = Path(r"C:\ieum")

SCOPE = "https://www.googleapis.com/auth/drive.file"

APP = "ieum_candidate2_archive_backup_v1"

FOLDER_MIME = "application/vnd.google-apps.folder"


def calculate_hashes(path: Path):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)

    return md5.hexdigest(), sha256.hexdigest()


def write_new_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)

    # 기존 report를 덮어쓰지 않음
    with path.open("x", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )

        f.flush()
        os.fsync(f.fileno())


def create_drive_service(token_path: Path):
    if token_path.name.lower() == "drive_readonly_token.json":
        raise ValueError(
            "Readonly Drive token cannot be used for upload."
        )

    if not token_path.exists():
        raise ValueError(
            f"Upload token not found: {token_path}"
        )

    token_obj = json.loads(
        token_path.read_text(encoding="utf-8")
    )

    if set(token_obj.get("scopes", [])) != {SCOPE}:
        raise ValueError(
            "Upload token must have exactly drive.file scope."
        )

    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    creds = Credentials.from_authorized_user_file(
        str(token_path),
        scopes=[SCOPE],
    )

    if not creds.valid:
        creds.refresh(Request())

    return build(
        "drive",
        "v3",
        credentials=creds,
        cache_discovery=False,
    )


def list_children(api, parent_id):
    items = []
    page_token = None

    while True:
        result = (
            api.files()
            .list(
                q=f"'{parent_id}' in parents and trashed=false",
                fields=(
                    "nextPageToken,"
                    "files("
                    "id,name,mimeType,size,"
                    "md5Checksum,parents,"
                    "appProperties,trashed"
                    ")"
                ),
                pageSize=1000,
                pageToken=page_token,
            )
            .execute(num_retries=5)
        )

        items.extend(result.get("files", []))

        page_token = result.get("nextPageToken")

        if not page_token:
            return items


def get_or_create_archive_folder(
    api,
    folder_name: str,
):
    props = {
        "backup_app": APP
    }

    matches = [
        item
        for item in list_children(api, "root")
        if (
            item["name"] == folder_name
            and item["mimeType"] == FOLDER_MIME
        )
    ]

    owned = [
        item
        for item in matches
        if item.get("appProperties") == props
    ]

    if len(owned) > 1:
        raise ValueError(
            "Duplicate app-owned archive folders found."
        )

    if owned:
        return owned[0]["id"]

    body = {
        "name": folder_name,
        "mimeType": FOLDER_MIME,
        "parents": ["root"],
        "appProperties": props,
    }

    result = (
        api.files()
        .create(
            body=body,
            fields="id",
        )
        .execute(num_retries=5)
    )

    return result["id"]


def upload_archive(
    args,
    archive_path: Path,
    local_size: int,
    local_md5: str,
    local_sha256: str,
    report: dict,
):
    from googleapiclient.http import MediaFileUpload

    api = create_drive_service(args.token)

    root_id = get_or_create_archive_folder(
        api,
        args.drive_folder,
    )

    # 같은 이름의 ZIP이 이미 있는지 확인
    existing = [
        item
        for item in list_children(api, root_id)
        if item["name"] == archive_path.name
    ]

    if len(existing) > 1:
        raise ValueError(
            "Duplicate remote archive filename."
        )

    # 이미 존재하고 검증까지 일치하면 재업로드하지 않음
    if existing:
        remote = existing[0]

        remote_size = int(
            remote.get("size", -1)
        )

        remote_md5 = remote.get(
            "md5Checksum"
        )

        if (
            remote_size == local_size
            and remote_md5 == local_md5
        ):
            report.update(
                status="COMPLETE",
                backup_verified=True,
                upload_performed=False,
                skipped_verified_existing=True,
                drive_root_id=root_id,
                drive_file_id=remote["id"],
                remote_size=remote_size,
                remote_md5=remote_md5,
            )

            return

        # 기존 파일이 있는데 내용이 다르면 덮어쓰지 않음
        raise ValueError(
            "Remote archive already exists "
            "but size/MD5 differs. "
            "Refusing overwrite."
        )

    properties = {
        "backup_app": APP,
        "batch_id": args.batch_id,
        "sha256": local_sha256,
    }

    media = MediaFileUpload(
        str(archive_path),
        mimetype="application/zip",
        chunksize=16 * 1024 * 1024,
        resumable=True,
    )

    try:
        request = (
            api.files()
            .create(
                body={
                    "name": archive_path.name,
                    "parents": [root_id],
                    "appProperties": properties,
                },
                media_body=media,
                fields=(
                    "id,name,size,"
                    "md5Checksum,appProperties"
                ),
            )
        )

        response = None

        while response is None:
            status, response = request.next_chunk(
                num_retries=5
            )

            if status:
                print(
                    f"Uploaded "
                    f"{status.progress() * 100:.1f}%",
                    flush=True,
                )

    finally:
        media.stream().close()

    # Drive 측 size / MD5 검증
    remote_size = int(
        response.get("size", -1)
    )

    remote_md5 = response.get(
        "md5Checksum"
    )

    if (
        remote_size != local_size
        or remote_md5 != local_md5
    ):
        raise ValueError(
            "Uploaded archive size/MD5 "
            "verification failed."
        )

    # 업로드 도중 로컬 ZIP이 변하지 않았는지 재검증
    md5_after, sha256_after = calculate_hashes(
        archive_path
    )

    if (
        md5_after != local_md5
        or sha256_after != local_sha256
    ):
        raise ValueError(
            "Local archive changed during upload."
        )

    report.update(
        status="COMPLETE",
        backup_verified=True,
        upload_performed=True,
        skipped_verified_existing=False,
        drive_root_id=root_id,
        drive_file_id=response["id"],
        remote_size=remote_size,
        remote_md5=remote_md5,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--batch-id",
        required=True,
    )

    parser.add_argument(
        "--archive-root",
        type=Path,
        default=(
            ROOT
            / "data"
            / "processed_candidate2"
        ),
    )

    parser.add_argument(
        "--token",
        type=Path,
        default=(
            ROOT
            / "credentials"
            / "drive_upload_token.json"
        ),
    )

    parser.add_argument(
        "--drive-folder",
        default="processed_candidate2_archives",
    )

    parser.add_argument(
        "--report-root",
        type=Path,
        default=(
            ROOT
            / "results"
            / "candidate2_drive_backup"
            / "archives"
        ),
    )

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
    )

    args = parser.parse_args()

    if not re.fullmatch(
        r"BATCH_(00[1-9]|01[0-9]|02[0-8])",
        args.batch_id,
    ):
        parser.error(
            "Choose BATCH_001..BATCH_028"
        )

    archive_path = (
        args.archive_root
        / f"{args.batch_id}.zip"
    ).resolve()

    if not archive_path.is_file():
        raise SystemExit(
            f"Archive not found: {archive_path}"
        )

    report_path = (
        args.report_root
        / (
            f"{args.batch_id}_archive_backup_"
            f"{uuid.uuid4().hex}.json"
        )
    )

    report = {
        "batch_id": args.batch_id,
        "archive": str(archive_path),
        "status": "FAILED",
        "backup_verified": False,
        "local_files_deleted": False,
        "mode": (
            "UPLOAD"
            if args.execute
            else "DRY_RUN"
        ),
    }

    start = time.perf_counter()

    try:
        local_size = archive_path.stat().st_size

        print(
            "Calculating local MD5/SHA256...",
            flush=True,
        )

        local_md5, local_sha256 = (
            calculate_hashes(archive_path)
        )

        report.update(
            archive_bytes=local_size,
            local_md5=local_md5,
            local_sha256=local_sha256,
        )

        print(
            f"Archive: {archive_path}",
            flush=True,
        )

        print(
            f"Size: {local_size:,} bytes",
            flush=True,
        )

        print(
            f"SHA256: {local_sha256}",
            flush=True,
        )

        if args.execute:
            print(
                "Starting Google Drive upload...",
                flush=True,
            )

            upload_archive(
                args,
                archive_path,
                local_size,
                local_md5,
                local_sha256,
                report,
            )

        else:
            report.update(
                status="DRY_RUN_VALIDATED",
                backup_verified=False,
                upload_performed=False,
                network_calls_permitted=False,
            )

    except Exception as exc:
        report.update(
            status="FAILED",
            backup_verified=False,
            error_type=type(exc).__name__,
            error=str(exc),
        )

    report["elapsed_sec"] = (
        time.perf_counter() - start
    )

    write_new_json(
        report_path,
        report,
    )

    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "Report:",
        report_path,
    )

    return (
        0
        if report["status"]
        in (
            "COMPLETE",
            "DRY_RUN_VALIDATED",
        )
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())