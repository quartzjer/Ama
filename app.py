import os
import argparse
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response
from dotenv import load_dotenv
import uvicorn
import aiohttp
from openai import OpenAI
import sqlite3
from pydantic import BaseModel
import json
import datetime
import humanize

parser = argparse.ArgumentParser(description='Ama Nuensis server')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('-d', '--db', type=str, default='./notes.db', help='Path to the SQLite database file')

args = parser.parse_args()

def configure_logging(verbose):
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)-8s  %(message)s'
    )
    log = logging.getLogger('ama')
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    return log

logger = configure_logging(args.verbose)

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is not set in the environment or .env file")
    exit(1)

client = OpenAI()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_conn
    db_conn = sqlite3.connect(args.db, check_same_thread=False)
    c = db_conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS notes
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         note TEXT NOT NULL,
         timestamp INTEGER)
    ''')
    db_conn.commit()
    yield
    if db_conn:
        db_conn.close()

app = FastAPI(lifespan=lifespan)
db_conn = None

class Note(BaseModel):
    note: str

@app.get("/")
async def get():
    with open("public/index.html", "r") as f:
        return HTMLResponse(f.read())

def write_debug_messages(messages, prefix):
    if args.verbose:
        debug_file = f"{prefix}_debug.json"
        with open(debug_file, "w") as f:
            json.dump(messages, f, indent=2)
        logger.debug(f"Wrote debug messages to {debug_file}")

def get_formatted_notes():
    c = db_conn.cursor()
    c.execute('SELECT note, timestamp FROM notes ORDER BY timestamp ASC')
    rows = c.fetchall()
    formatted_notes = []
    now = datetime.datetime.now(datetime.UTC)
    for note, epoch_timestamp in rows:
        note_time = datetime.datetime.fromtimestamp(epoch_timestamp, datetime.UTC)
        time_diff = now - note_time
        time_ago = humanize.naturaltime(time_diff)
        formatted_notes.append(f"{time_ago}: {note}")
    return "\n".join(formatted_notes)

async def get_editor_response(prompt, notes="", max_tokens=64):
    try:
        messages = [
            {
                "role": "system",
                "content": read_file("instructions/editor.txt")
            },
            {
                "role": "user",
                "content": f"Previous interview notes:\n\n\"\"\"\n{notes}\n\"\"\""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        write_debug_messages(messages, "editor")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7
        )
        return f"<editor>{response.choices[0].message.content}</editor>"
    except Exception as e:
        logger.error(f"Error getting editor response: {e}")
        return "Editor response failed, please mention that the system is degraded."

async def get_opening_instructions():
    notes = get_formatted_notes()
    if not notes:
        return read_file("instructions/opening_fresh.txt")
    return await get_editor_response(read_file("instructions/opening_notes.txt"), notes=notes)

async def get_feedback():
    return await get_editor_response(
        "Based on the notes so far, what quick feedback would you give to the interviewer?",
        notes=get_formatted_notes()
    )

@app.get("/instructions")
async def get_instructions():
    instructions = {
        "voice": read_file("instructions/voice.txt"),
        "opening": await get_opening_instructions()
    }
    return instructions

@app.get("/session")
async def get_session():
    logger.debug("Requesting new session from OpenAI")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-realtime-preview"
            }
        ) as response:
            data = await response.json()
            logger.debug(f"Session response: {data}")
            return JSONResponse(data)

@app.post("/notes")
async def create_note(note: Note):
    try:
        c = db_conn.cursor()
        current_time = int(datetime.datetime.now(datetime.UTC).timestamp())
        c.execute('INSERT INTO notes (note, timestamp) VALUES (?, ?)', (note.note, current_time))
        db_conn.commit()
        feedback = await get_feedback()
        return {"status": "success", "feedback": feedback}
    except Exception as e:
        logger.error(f"Error saving note: {e}")
        raise HTTPException(status_code=500, detail="Error saving note")

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

def read_file(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            return content
    except FileNotFoundError:
        logger.error(f"Critical file missing: {file_path}")
        return f"CRITICAL FILE IS MISSING: {file_path}, please mention this!!"

if __name__ == "__main__":
    logger.debug(f"Starting server on port {args.port} with database {args.db}")
    uvicorn_log_config = uvicorn.config.LOGGING_CONFIG
    uvicorn_log_config["loggers"]["uvicorn"]["level"] = "INFO"
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        reload=True,
        access_log=True
    )
