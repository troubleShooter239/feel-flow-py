from os import path

from pydantic import BaseModel
from pydantic_settings import BaseSettings


class FacialAttributes(BaseModel):
    age: str = "age.h5"
    emotion: str = "emotion.h5"
    gender: str = "gender.h5"
    race: str = "race.h5"


class RecognitionAttributes(BaseModel):
    arcface: str = "arcface.h5"
    deepid: str = "deepid.h5"
    facenet: str = "facenet.h5"
    facenet512: str = "facenet512.h5"
    openface: str = "openface.h5"
    vggface: str = "vgg_face.h5"
    

class ModelsSettings(BaseSettings):
    download_folder: str = path.expanduser('~') + "/.feel-flow"
    download_url: str = "https://github.com/troubleShooter239/feel-flow-py/releases/download/v1.0.0/"
    face_attrs: FacialAttributes = FacialAttributes()
    recognition_attrs: RecognitionAttributes = RecognitionAttributes()
    
models_settings = ModelsSettings()
    