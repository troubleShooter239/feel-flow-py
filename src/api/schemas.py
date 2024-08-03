from pydantic import BaseModel


class Analyze(BaseModel):
    """Pydantic model for image analysis request."""
    b64_img: str = "data:image/jpeg;base64"
    actions: dict[str, bool] = {
        "age": True, "emotion": True, "gender": True, "race": True
    }


class Metadata(BaseModel):
    """Pydantic model for image metadata request."""
    b64_img: str = "Base64 Encoded String"


class Verify(BaseModel):
    """Pydantic model for face verification request."""
    b64_img1: str = "data:image/jpeg;base64"
    b64_img2: str = "data:image/jpeg;base64"
    r_model_name: str = "vgg-face"
    distance_metric: str = "cosine"
