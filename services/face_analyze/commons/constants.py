_DEEPFACE_HOME: str = "DEEPFACE_HOME"
_DOWNLOAD_URL: str = "https://github.com/serengil/deepface_models/releases/download/v1.0/"
_WEIGHTS: str = "/.deepface/weights/"

# face attributes
AGE_NAME: str = "age_model_weights.h5"
DOWNLOAD_URL_AGE: str = _DOWNLOAD_URL + AGE_NAME
PATH_WEIGHTS_AGE: str = _WEIGHTS + AGE_NAME

EMOTION_NAME: str = "facial_expression_model_weights.h5"
DOWNLOAD_URL_EMOTION: str = _DOWNLOAD_URL + EMOTION_NAME
PATH_WEIGHTS_EMOTION: str = _WEIGHTS + EMOTION_NAME

GENDER_NAME: str = "gender_model_weights.h5"
DOWNLOAD_URL_GENDER: str = _DOWNLOAD_URL + GENDER_NAME
PATH_WEIGHTS_GENDER: str = _WEIGHTS + GENDER_NAME

RACE_NAME: str = "race_model_single_batch.h5"
DOWNLOAD_URL_RACE: str = _DOWNLOAD_URL + RACE_NAME
PATH_WEIGHTS_RACE: str = _WEIGHTS + RACE_NAME

# recognition
ARCFACE_NAME: str = "arcface_weights.h5"
DOWNLOAD_URL_ARCFACE: str = _DOWNLOAD_URL + ARCFACE_NAME
PATH_WEIGHTS_ARCFACE: str = _WEIGHTS + ARCFACE_NAME

DEEPFACE_NAME: str = "VGGFace2_DeepFace_weights_val-0.9034.h5"
DOWNLOAD_URL_DEEPFACE: str = f"https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/{DEEPFACE_NAME}.zip"
PATH_WEIGHTS_DEEPFACE: str = _WEIGHTS + DEEPFACE_NAME

DEEPID_NAME: str = "deepid_keras_weights.h5"
DOWNLOAD_URL_DEEPID: str = _DOWNLOAD_URL + DEEPID_NAME
PATH_WEIGHTS_DEEPID: str = _WEIGHTS + DEEPID_NAME

DLIB_NAME: str = "dlib_face_recognition_resnet_model_v1.dat"
DOWNLOAD_URL_DLIB: str = _DOWNLOAD_URL + DLIB_NAME
PATH_WEIGHTS_DLIB: str = _WEIGHTS + DLIB_NAME

FACENET_NAME: str = "facenet_weights.h5"
DOWNLOAD_URL_FACENET: str = _DOWNLOAD_URL + FACENET_NAME
PATH_WEIGHTS_FACENET: str = _WEIGHTS + FACENET_NAME

FACENET512_NAME: str = "facenet512_weights.h5"
DOWNLOAD_URL_FACENET512: str = _DOWNLOAD_URL + FACENET512_NAME
PATH_WEIGHTS_FACENET512: str = _WEIGHTS + FACENET512_NAME

OPENFACE_NAME: str = "openface_weights.h5"
DOWNLOAD_URL_OPENFACE: str = _DOWNLOAD_URL + OPENFACE_NAME
PATH_WEIGHTS_OPENFACE: str = _WEIGHTS + OPENFACE_NAME

SFACE_NAME: str = "face_recognition_sface_2021dec.onnx"
DOWNLOAD_URL_SFACE: str = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/" + SFACE_NAME
PATH_WEIGHTS_SFACE: str = _WEIGHTS + SFACE_NAME

VGGFACE_NAME: str = "vgg_face_weights.h5"
DOWNLOAD_URL_VGGFACE: str = _DOWNLOAD_URL + VGGFACE_NAME
PATH_WEIGHTS_VGGFACE: str = _WEIGHTS + VGGFACE_NAME
