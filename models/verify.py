from pydantic import BaseModel


class VerifyModel(BaseModel):
    b64_img1: str
    b64_img2: str
    model_name: str
    distance_metric: str
