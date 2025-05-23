from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
from loguru import logger
from config import settings
from api.endpoints import router

# Configure logging
logger.add(
    settings.log_dir / settings.log_file,
    rotation="500 MB",
    retention="10 days",
    level=settings.log_level
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting KTRU RAG Classification System")

    # Initialize services
    try:
        from services.embeddings import get_embedding_service
        from services.vector_store import get_vector_store
        from services.llm_service import get_llm_service
        from services.classifier import get_classifier_service

        # Pre-initialize all services
        get_embedding_service()
        get_vector_store()
        get_llm_service()
        get_classifier_service()

        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

    yield

    logger.info("Shutting down KTRU RAG Classification System")


# Create FastAPI app
app = FastAPI(
    title="KTRU RAG Classification API",
    description="API for classifying products to KTRU codes using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router, prefix="/api/v1", tags=["classification"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "KTRU RAG Classification API",
        "version": "1.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        reload=False  # Set to True for development
    )