from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.mongodb import connect_to_mongo, close_mongo_connection
from app.routes.auth_routes import router as auth_router
from app.routes.submission_routes import router as submission_router
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    await connect_to_mongo()
    yield
    # Shutdown actions
    await close_mongo_connection()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Scalable ML Platform API for Requirements Submission.",
    version="1.1.0",
    lifespan=lifespan
)

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(submission_router, prefix="/submit", tags=["Submissions"])

@app.get("/")
async def root():
    return {"message": "Welcome to the simplified ML Platform API"}
