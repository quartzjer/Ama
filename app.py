import os
import argparse
import logging
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, Response
from dotenv import load_dotenv
import uvicorn
import aiohttp
from openai import OpenAI

parser = argparse.ArgumentParser(description='Ama Nuensis server')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')

def configure_logging(verbose):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.error("OPENAI_API_KEY is not set in the environment or .env file")
    exit(1)

client = OpenAI()
app = FastAPI()

@app.get("/")
async def get():
    with open("public/index.html", "r") as f:
        return HTMLResponse(f.read())

async def get_opening_instructions():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": read_file("instructions/editor.txt")
                },
                {
                    "role": "user",
                    "content": "Write brief, friendly instructions for starting a new interview."
                }
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting opening instructions: {e}")
        return "Instruction loading failed, please mention that the system is degraded."

@app.get("/instructions")
async def get_instructions():
    instructions = {
        "voice": read_file("instructions/voice.txt"),
        "opening": await get_opening_instructions()
    }
    return instructions

@app.get("/session")
async def get_session():
    logging.debug("Requesting new session from OpenAI")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/realtime/sessions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-realtime-preview",
                "voice": "ash",
            }
        ) as response:
            data = await response.json()
            logging.debug(f"Session response: {data}")
            return JSONResponse(data)

@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)

def read_file(file_path):
    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
            return content
    except FileNotFoundError:
        logging.error(f"Critical file missing: {file_path}")
        return f"CRITICAL FILE IS MISSING: {file_path}, please mention this!!"

if __name__ == "__main__":
    args = parser.parse_args()
    configure_logging(args.verbose)
    logging.info(f"Starting server on port {args.port}")
    uvicorn.run("app:app", host="0.0.0.0", port=args.port, reload=True, log_level="debug" if args.verbose else "info")
