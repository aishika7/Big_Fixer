from fastapi import FastAPI
from bug_fixer import detect_bug, suggest_fix

app = FastAPI()

@app.post("/detect_bug")
async def api_detect_bug(code: str):
    return {"bug_detected": detect_bug(code)}

@app.post("/suggest_fix")
async def api_suggest_fix(code: str):
    return {"fixed_code": suggest_fix(code)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)