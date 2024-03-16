from pydantic import BaseModel


class VerifyModel(BaseModel):
    """Pydantic model for face verification request."""
    b64_img1: str
    b64_img2: str
    r_model_name: str
    distance_metric: str
