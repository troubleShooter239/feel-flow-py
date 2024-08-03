from pydantic import BaseModel


class Region(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Emotion(BaseModel):
    angry: float
    disgust: float
    fear: float
    happy: float
    sad: float
    surprise: float
    neutral: float


class Gender(BaseModel):
    woman: float
    man: float


class Race(BaseModel):
    asian: float
    indian: float
    black: float
    white: float
    middle_eastern: float
    latino_hispanic: float


class Actions(BaseModel):
    age: int
    emotion: Emotion
    dominant_emotion: str
    gender: Gender
    dominant_gender: str
    race: Race
    dominant_race: str


class FaceAnalysis(BaseModel):
    region: Region
    face_confidence: float
    actions: Actions
    
class FacialAreas(BaseModel):
    img1: Region
    img2: Region


class Summary(BaseModel):
    BBox: tuple[int, int, int, int] | None = None
    BandNames: tuple[str, ...]
    DateTime: str | None = None
    ExifOffset: int | None = None
    Extrema: tuple[float, float] | tuple[tuple[int, int], ...]
    FileType: str
    FormatDescription: str
    HasTransparency: bool
    HostComputer: str | None = None
    ImageSize: tuple[int, int]
    Make: str | None
    Megapixels: float | None = None
    Mime: str | None = None
    Mode: str
    Model: str | None = None
    Readonly: bool
    Software: str | None = None
    XResolution: float | None = None
    YCbCrPositioning: int | None = None
    YResolution: float | None = None
        

class Ifd1(BaseModel):
    Compression: int | None = None
    JpegIFByteCount: int | None = None
    JpegIFOffset: int| None = None
    ResolutionUnit: int | None = None
    XResolution: float | None = None 
    YResolution: float | None = None


class ExifData(BaseModel):
    ApertureValue: float | None = None
    BrightnessValue: float | None = None
    ColorSpace: int | None = None 
    ComponentsConfiguration: bytes | None = None
    CompositeImage: int | None = None
    DateTimeDigitized: str | None = None
    DateTimeOriginal: str | None = None
    ExifImageHeight: int | None = None
    ExifImageWidth: int | None = None
    ExifVersion: bytes | None = None
    ExposureBiasValue: float | None
    ExposureMode: int | None = None
    ExposureProgram: int | None = None
    ExposureTime: float | None = None
    FNumber: float | None = None
    Flash: int | None = None
    FlashPixVersion: bytes | None = None
    FocalLength: float | None = None
    FocalLengthIn35mmFilm: int | None = None
    ISOSpeedRatings: int | None = None
    LensMake: str | None = None
    LensModel: str | None = None
    LensSpecification: tuple[float, float, float, float] | None = None
    MeteringMode: int | None = None
    OffsetTime: str | None = None
    OffsetTimeDigitized: str | None = None
    OffsetTimeOriginal: str | None = None
    SceneCaptureType: int | None = None
    SceneType: bytes | None = None
    SensingMethod: int | None = None
    ShutterSpeedValue: float | None = None
    SubjectLocation: tuple[int, int, int, int] | None = None
    SubsecTimeDigitized: int | None = None
    SubsecTimeOriginal: int | None = None
    WhiteBalance: int | None = None

class GpsInfo(BaseModel):
    GPSAltitude: float | None = None
    GPSAltitudeRef: bytes | None = None
    GPSDateStamp: str | None = None
    GPSDestBearing: float | None = None
    GPSDestBearingRef: str | None = None
    GPSHPositioningError: float | None = None
    GPSImgDirection: float | None = None
    GPSImgDirectionRef: str | None = None
    GPSLatitude: tuple[float, float, float] | None = None
    GPSLatitudeRef: str | None = None
    GPSLongitude: tuple[float, float, float] | None = None
    GPSLongitudeRef: str | None = None
    GPSSpeed: float | None = None
    GPSSpeedRef: str | None = None
    GPSTimeStamp: tuple[float, float, float] | None = None


class AnalyzeResult(BaseModel):
    faces: list[FaceAnalysis]
    
    
class MetadataResult(BaseModel):
    Summary: Summary
    Exif: ExifData | None = None
    GPSInfo: GpsInfo | None = None
    IFD1: Ifd1 | None = None


class VerifyResult(BaseModel):
    verified: bool
    distance: float
    threshold: float
    model: str
    distance_metric: str
    facial_areas: FacialAreas
