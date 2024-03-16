from pydantic import BaseModel


class MetadataModel(BaseModel):
    """Pydantic model for image metadata request."""
    b64_img: str
