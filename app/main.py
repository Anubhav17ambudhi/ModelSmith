from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.database.mongodb import connect_to_mongo, close_mongo_connection
from app.routes.auth_routes import router as auth_router
from app.routes.model_routes import router as model_router
from app.routes.training_routes import router as training_router
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
    description="Scalable ML Platform API enabling users to train custom neural networks asynchronously.",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(model_router, prefix="/models", tags=["Model Configurations"])
app.include_router(training_router, prefix="/training", tags=["Training Jobs"])

@app.get("/")
async def root():
    return {"message": "Welcome to the ML Platform API"}
