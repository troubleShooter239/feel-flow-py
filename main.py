from fastapi import FastAPI
from uvicorn import run

from routers.api import router

app = FastAPI()
app.include_router(router)
# @app.get("/")
# def read_root():
#     return {"Hello": "World"}

# @app.post("/")
# def get_post():
#     global counter_int
#     counter_int += 1
#     return {"counter": counter_int}


if __name__ == "__main__":
    run("main:app", reload=True)