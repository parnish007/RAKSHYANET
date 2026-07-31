"""RakshyaNet FastAPI launcher."""
import sys
from pathlib import Path

import uvicorn


project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


if __name__ == "__main__":
    print("Starting RakshyaNet Backend Server...")
    print("API: http://localhost:8000")
    print("WebSocket: ws://localhost:8000/ws")
    print("Docs: http://localhost:8000/docs")
    print("\nPress CTRL+C to stop\n")

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
