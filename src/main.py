from fastapi import FastAPI
from uvicorn import run

from api import image

app = FastAPI()
app.include_router(image.router)

if __name__ == "__main__":
    run("main:app", reload=True)