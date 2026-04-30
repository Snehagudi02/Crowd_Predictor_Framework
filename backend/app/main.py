from fastapi import FastAPI
from backend.app.api.routes import router

app = FastAPI(title="Crowd Monitoring API")
app.include_router(router)
