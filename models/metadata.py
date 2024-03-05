from pydantic import BaseModel


class MetadataModel(BaseModel):
    b64_img: str
