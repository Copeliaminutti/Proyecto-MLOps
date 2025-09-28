from fastapi import FastAPI

app = FastAPI(title="ML Service")

@app.get("/health")
def health():
    return {"status": "ok"}
