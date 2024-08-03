from io import BytesIO

from cv2 import COLOR_BGR2GRAY, resize, cvtColor
from numpy import ndarray, expand_dims, float32, pad, argmin
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS, GPSTAGS, IFD
from tensorflow.keras.preprocessing import image

from .models.response_models import Actions, FaceAnalysis, Ifd1, AnalyzeResult, ExifData, FacialAreas, GpsInfo, MetadataResult, Region, Summary, VerifyResult
from .detectors.opencv_client import DetectorWrapper
from .loaders.image_loader import load_image
from .commons.distance import find_cosine, find_euclidean
from .commons.face_processor import FaceProcessor
from .commons.functions import build_model, find_size, normalize_input, find_threshold


def extract_faces(img: str | ndarray, target_size: tuple = (224, 224), grayscale: bool = False, 
                  enforce_detection: bool = True, align: bool = True) -> tuple[tuple[ndarray, dict[str, int], float]]:
    """Extract faces from an image.

    Args:
        img (Union[str, np.ndarray]): Path to the image or image array.
        target_size (tuple, optional): Target size of the face. Defaults to (224, 224).
        grayscale (bool, optional): Whether to convert the image to grayscale. Defaults to False.
        enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
        align (bool, optional): Whether to align faces. Defaults to True.

    Returns:
        List[Tuple[np.ndarray, Dict[str, int], float]]: List of tuples containing face images, 
        facial area coordinates, and confidence scores."""
    img, img_name = load_image(img)

    face_objs = DetectorWrapper.detect_faces(img, align)

    if len(face_objs) == 0 and enforce_detection:
        raise ValueError(
            f"Face could not be detected in {img_name or None}."
            "Please confirm that the picture is a face photo "
            "or consider to set enforce_detection param to False."
        )
    
    img_region = [0, 0, img.shape[1], img.shape[0]]
    if len(face_objs) == 0 and not enforce_detection:
        face_objs = ((img, img_region, 0))

    extracted_faces = []
    for current_img, reg, confidence in face_objs:
        if current_img.shape[0] < 0 or current_img.shape[1] < 0:
            continue

        if grayscale:
            current_img = cvtColor(current_img, COLOR_BGR2GRAY)

        factor = min(target_size[0] / current_img.shape[0], target_size[1] / current_img.shape[1])

        current_img = resize(current_img, (int(current_img.shape[1] * factor), int(current_img.shape[0] * factor)))

        diff_0 = target_size[0] - current_img.shape[0]
        diff_1 = target_size[1] - current_img.shape[1]
        
        if grayscale:
            current_img = pad(current_img, 
                              ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), "constant")
        else:
            current_img = pad(current_img, 
                              ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), "constant")

        if current_img.shape[0:2] != target_size:
            current_img = resize(current_img, target_size)
            
        img_pixels = expand_dims(image.img_to_array(current_img), axis=0) / 255
        regs = {"x": int(reg[0]), "y": int(reg[1]), "w": int(reg[2]), "h": int(reg[3])}
        extracted_faces.append((img_pixels, regs, confidence))

    if len(extracted_faces) == 0 and enforce_detection:
        raise ValueError(f"Detected face shape is {img.shape}. Consider to set enforce_detection arg to False.")

    return tuple(extracted_faces)


def represent(img_path: str | ndarray, model_name: str = "vgg_face", 
              enforce_detection: bool = True, detector_backend: str = "opencv", 
              align: bool = True, normalization: str = "base") -> tuple[dict[str, list[float] | dict[str, int] | float]]:
    """Represent an image with the specified recognition model.

    Args:
        img_path (Union[str, np.ndarray]): Path to the image or image array.
        model_name (str, optional): Name of the recognition model. Defaults to "VGG-Face".
        enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
        detector_backend (str, optional): Backend for face detection. Defaults to "opencv".
        align (bool, optional): Whether to align faces. Defaults to True.
        normalization (str, optional): Type of normalization. Defaults to "base".

    Returns:
        list[dict[str, list[float] | dict[str, int] | float]]: List of representations."""
    model = build_model(model_name)
    target_size = find_size(model_name)
    if detector_backend != "skip":
        img_objs = extract_faces(img_path, (target_size[1], target_size[0]), False, enforce_detection, align)
    else:
        img, _ = load_image(img_path)
        if len(img.shape) == 4:
            img = img[0]
        if len(img.shape) == 3:
            img = resize(img, target_size)
            img = expand_dims(img, axis=0)
            if img.max() > 1:
                img = (img.astype(float32) / 255.0).astype(float32)

        img_objs = [(img, {"x": 0, "y": 0, "w": img.shape[1], "h": img.shape[2]}, 0.0)]
    return tuple([{
        "embedding": model.find_embeddings(normalize_input(i, normalization)),
        "facial_area": r,
        "face_confidence": c
    } for i, r, c in img_objs])



def analyze(img: str | ndarray, actions: dict[str, bool] = {"age": True, "emotion": True, "gender": True, "race": True}, 
            align: bool = True, enforce_detection: bool = True) -> AnalyzeResult:
    """Analyze an image to detect faces and perform specified actions.

    Args:
        img (Union[str, np.ndarray]): Input image path or array.
        actions (Dict[str, bool], optional): Dictionary specifying which actions to perform. Defaults to {"age": True, "emotion": True, "gender": True, "race": True}.
        align (bool, optional): Whether to perform face alignment. Defaults to True.
        enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.

    Returns:
        AnalyzeResult: List of dictionaries containing analysis results for each detected face."""
    img_objs = extract_faces(img, (224, 224), False, enforce_detection, align)
    models = {action: build_model(action) for action, selected in actions.items() if selected}
    
    resp_objects = []
    for img, region, confidence in img_objs:
        if img.shape[0] <= 0 or img.shape[1] <= 0: 
            continue
        
        obj = {}
        for a, m in models.items():
            obj.update(getattr(FaceProcessor, a)(m.predict(img)))
        resp_objects.append(FaceAnalysis(
            region=Region(**region), 
            face_confidence=confidence, 
            actions=Actions(**obj)
        ))

    return AnalyzeResult(faces=resp_objects)


def get_image_metadata(image: bytes) -> MetadataResult:
    """Extract metadata from an image.

    Args:
        image (bytes): Image bytes.

    Returns:
        MetadataResult: Image metadata."""
    try:
        i = Image.open(BytesIO(image))
    except UnidentifiedImageError:
        raise ValueError("Not correct format of image(must be Base64 Encoded String)")
    
    summary_data = {
        "BBox": i.getbbox(),
        "BandNames": i.getbands(),
        "DateTime": None,
        "ExifOffset": None,
        "Extrema": i.getextrema(),
        "FileType": i.format,
        "FormatDescription": i.format_description,
        "HasTransparency": i.has_transparency_data,
        "HostComputer": None,
        "ImageSize": i.size,
        "Make": None,
        "Megapixels": round(i.size[0] * i.size[1] / 1000000, 2),
        "Mime": Image.MIME.get(i.format),
        "Mode": i.mode,
        "Model": None,
        "Readonly": not not i.readonly,
        "Software": None,
        "XResolution": None,
        "YCbCrPositioning": None,
        "YResolution": None
    }
    
    exif = i.getexif()
    i.close()
    if not exif:
        return MetadataResult(Summary=Summary(**summary_data))
    
    for k, v in exif.items():
        try:
            tag = TAGS[k]
            if tag in summary_data.keys():
                summary_data[tag] = v
        except KeyError:
            continue

    exif_data = None
    if ifd_exif := exif.get_ifd(IFD.Exif):
        exif_data = ExifData(**{
            TAGS[k]: v for k, v in ifd_exif.items() if TAGS[k] in ExifData.model_fields.keys()
        })
    
    gps_data = None
    if ifd_gps := exif.get_ifd(IFD.GPSInfo):
        gps_data = GpsInfo(**{
            GPSTAGS[k]: v for k, v in ifd_gps.items() if GPSTAGS[k] in GpsInfo.model_fields.keys()
        })
        
    ifd1_data = None
    if ifd_ifd1 := exif.get_ifd(IFD.IFD1):
        ifd1_data = Ifd1(**{
            TAGS[k]: v for k, v in ifd_ifd1.items() if TAGS[k] in Ifd1.model_fields.keys()
        })
    
    return MetadataResult(Summary=Summary(**summary_data), Exif=exif_data, 
                          GPSInfo=gps_data, IFD1=ifd1_data)


def verify(img1: str | ndarray, img2: str | ndarray, model_name: str = "vgg_face", 
           distance_metric: str = "cosine", enforce_detection: bool = True, 
           align: bool = True, normalization: str = "base") -> VerifyResult:
    """Verify whether two images contain the same person.

    Args:
        img1 (Union[str, np.ndarray]): Path to or array of the first image.
        img2 (Union[str, np.ndarray]): Path to or array of the second image.
        model_name (str, optional): Model to be used for facial recognition. Defaults to "VGG-Face".
        distance_metric (str, optional): Distance metric to measure similarity. Defaults to "cosine".
        enforce_detection (bool, optional): Whether to enforce face detection. Defaults to True.
        align (bool, optional): Whether to align faces. Defaults to True.
        normalization (str, optional): Type of normalization to be applied. Defaults to "base".

    Returns:
        VerifyResult: Verification result."""
    target_size = find_size(model_name)

    distances, regions = [], []
    for c1, r1, _ in extract_faces(img1, target_size, False, enforce_detection, align):
        for c2, r2, _ in extract_faces(img2, target_size, False, enforce_detection, align):
            repr1 = represent(c1, model_name, enforce_detection, "skip", align, normalization)[0]["embedding"]
            repr2 = represent(c2, model_name, enforce_detection, "skip", align, normalization)[0]["embedding"]

            if distance_metric == "cosine":
                dst = find_cosine(repr1, repr2)
            elif distance_metric == "euclidean":
                dst = find_euclidean(repr1, repr2)
            else:
                dst = find_euclidean(dst.l2_normalize(repr1), dst.l2_normalize(repr2))

            distances.append(dst)
            regions.append((r1, r2))

    threshold = find_threshold(model_name, distance_metric)
    distance = min(distances)
    facial_areas = regions[argmin(distances)]
    return VerifyResult(verified=True if distance <= threshold else False, distance=float(distance), 
                        threshold=threshold, model=model_name, distance_metric=distance_metric, 
                        facial_areas=FacialAreas(img1=Region(**facial_areas[0]), 
                                                 img2=Region(**facial_areas[1])))
