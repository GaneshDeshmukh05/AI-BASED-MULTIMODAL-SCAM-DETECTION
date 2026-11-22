import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import os

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add the main project directory to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from fusion import analyze_multimodal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ScamShield AI API",
    description="Multimodal scam and phishing detection API",
    version="1.1.0",
)

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BACKEND_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

DATABASE_PATH = BACKEND_DIR / "scamshield.db"
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg"}
ALLOWED_AUDIO_TYPES = {"audio/wav", "audio/x-wav", "audio/wave"}


def get_db_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                verdict TEXT,
                risk_level TEXT,
                overall_risk_score REAL,
                overall_confidence REAL,
                inputs_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at DESC)"
        )
        connection.commit()


initialize_database()


def save_scan(scan_id: str, created_at: str, inputs: dict, result: dict):
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO scans (
                id, created_at, verdict, risk_level,
                overall_risk_score, overall_confidence,
                inputs_json, result_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                created_at,
                result.get("verdict"),
                result.get("risk_level"),
                result.get("overall_risk_score"),
                result.get("overall_confidence"),
                json.dumps(inputs, ensure_ascii=False, default=str),
                json.dumps(result, ensure_ascii=False, default=str),
            ),
        )
        connection.commit()


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "ScamShield AI",
        "database": "connected",
    }


async def save_upload(
    upload: Optional[UploadFile],
    allowed_types: set,
    allowed_extensions: set,
) -> Optional[str]:
    if upload is None:
        return None

    extension = Path(upload.filename or "").suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file extension.",
        )

    if upload.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type.",
        )

    filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / filename
    size = 0

    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 15 MB limit.",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return str(destination)


@app.post("/api/scan")
async def scan_multimodal(
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    screenshot_file: Optional[UploadFile] = File(None),
    offline_mode: bool = Form(True),
):
    clean_text = text.strip() if text and text.strip() else None
    clean_url = url.strip() if url and url.strip() else None

    if not any([clean_url, clean_text, audio_file, screenshot_file]):
        raise HTTPException(
            status_code=400,
            detail="Please provide text, a URL, an image, or an audio file.",
        )

    audio_path = None
    screenshot_path = None

    try:
        screenshot_path = await save_upload(
            screenshot_file,
            ALLOWED_IMAGE_TYPES,
            {".png", ".jpg", ".jpeg"},
        )
        audio_path = await save_upload(
            audio_file,
            ALLOWED_AUDIO_TYPES,
            {".wav"},
        )

        logger.info("Starting multimodal scan")

        result = analyze_multimodal(
            audio_path=audio_path,
            url=clean_url,
            text=clean_text,
            screenshot_path=screenshot_path,
            include_domain_intelligence=not offline_mode,
            explainer_provider="deterministic",
        )

        result_data = result.to_dict()
        scan_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        # Store filenames only, not uploaded file contents.
        input_summary = {
            "text": clean_text,
            "url": clean_url,
            "audio_filename": audio_file.filename if audio_file else None,
            "screenshot_filename": screenshot_file.filename if screenshot_file else None,
            "offline_mode": offline_mode,
        }

        save_scan(scan_id, created_at, input_summary, result_data)

        response_data = {
            **result_data,
            "scan_id": scan_id,
            "created_at": created_at,
        }
        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Scan failed")
        raise HTTPException(
            status_code=500,
            detail="The analysis service encountered an unexpected error.",
        ) from exc
    finally:
        for path in (audio_path, screenshot_path):
            if path:
                Path(path).unlink(missing_ok=True)


@app.get("/api/history")
async def get_scan_history(limit: int = Query(default=50, ge=1, le=100)):
    try:
        with get_db_connection() as connection:
            rows = connection.execute(
                """
                SELECT id, created_at, verdict, risk_level,
                       overall_risk_score, overall_confidence, inputs_json
                FROM scans
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        items = []
        for row in rows:
            items.append(
                {
                    "scan_id": row["id"],
                    "created_at": row["created_at"],
                    "verdict": row["verdict"],
                    "risk_level": row["risk_level"],
                    "overall_risk_score": row["overall_risk_score"],
                    "overall_confidence": row["overall_confidence"],
                    "inputs": json.loads(row["inputs_json"]),
                }
            )

        return {"total": len(items), "scans": items}
    except Exception as exc:
        logger.exception("Could not retrieve scan history")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve scan history.",
        ) from exc


@app.get("/api/history/{scan_id}")
async def get_scan_details(scan_id: str):
    try:
        with get_db_connection() as connection:
            row = connection.execute(
                """
                SELECT id, created_at, inputs_json, result_json
                FROM scans
                WHERE id = ?
                """,
                (scan_id,),
            ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Scan not found.")

        return {
            "scan_id": row["id"],
            "created_at": row["created_at"],
            "inputs": json.loads(row["inputs_json"]),
            "result": json.loads(row["result_json"]),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Could not retrieve scan details")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve scan details.",
        ) from exc



@app.get("/api/dashboard/stats")
async def get_dashboard_stats():
    try:
        with get_db_connection() as connection:

            # Overall scan statistics
            totals = connection.execute("""
                SELECT
                    COUNT(*) AS total_scans,
                    SUM(CASE
                        WHEN UPPER(verdict) IN ('SCAM', 'MALICIOUS', 'PHISHING')
                        THEN 1 ELSE 0
                    END) AS scam_detections,
                    SUM(CASE
                        WHEN UPPER(verdict) IN ('SAFE', 'LEGITIMATE')
                        THEN 1 ELSE 0
                    END) AS safe_detections,
                    AVG(overall_risk_score) AS average_risk
                FROM scans
            """).fetchone()

            # Distribution by verdict
            distribution_rows = connection.execute("""
                SELECT verdict, COUNT(*) AS count
                FROM scans
                GROUP BY verdict
                ORDER BY count DESC
            """).fetchall()

            # Last seven days of scan activity
            daily_rows = connection.execute("""
                SELECT
                    substr(created_at, 1, 10) AS scan_date,
                    COUNT(*) AS total,
                    SUM(CASE
                        WHEN UPPER(verdict) IN ('SCAM', 'MALICIOUS', 'PHISHING')
                        THEN 1 ELSE 0
                    END) AS threats
                FROM scans
                WHERE created_at >= datetime('now', '-6 days')
                GROUP BY substr(created_at, 1, 10)
                ORDER BY scan_date ASC
            """).fetchall()

            # Five most recent scans
            recent_rows = connection.execute("""
                SELECT
                    id, created_at, verdict, risk_level,
                    overall_risk_score, overall_confidence,
                    inputs_json
                FROM scans
                ORDER BY created_at DESC
                LIMIT 5
            """).fetchall()

        # Fill missing days with zero values
        daily_lookup = {
            row["scan_date"]: row
            for row in daily_rows
            if row["scan_date"]
        }

        daily_activity = []

        from datetime import timedelta

        today = datetime.now(timezone.utc).date()

        for offset in range(6, -1, -1):
            day = today - timedelta(days=offset)
            key = day.isoformat()
            row = daily_lookup.get(key)

            daily_activity.append({
                "date": key,
                "scans": row["total"] if row else 0,
                "threats": row["threats"] if row else 0,
            })

        recent_scans = []

        for row in recent_rows:
            recent_scans.append({
                "scan_id": row["id"],
                "created_at": row["created_at"],
                "verdict": row["verdict"],
                "risk_level": row["risk_level"],
                "overall_risk_score": row["overall_risk_score"],
                "overall_confidence": row["overall_confidence"],
                "inputs": json.loads(row["inputs_json"]),
            })

        return {
            "total_scans": totals["total_scans"] or 0,
            "scam_detections": totals["scam_detections"] or 0,
            "safe_detections": totals["safe_detections"] or 0,
            "average_risk": round(totals["average_risk"] or 0, 4),
            "verdict_distribution": [
                {
                    "verdict": row["verdict"] or "UNKNOWN",
                    "count": row["count"],
                }
                for row in distribution_rows
            ],
            "daily_activity": daily_activity,
            "recent_scans": recent_scans,
        }

    except Exception as exc:
        logger.exception("Could not retrieve dashboard statistics")
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve dashboard statistics.",
        ) from exc
