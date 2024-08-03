# feel-flow-py

Feel Flow is a `RESTful API` designed for analyzing photos. The API is built on FastAPI and UVicorn, providing endpoints for various functionalities related to media analysis, particularly focusing on detecting human faces within images. The primary advantage of Feel Flow is its `high performance`, making it suitable for applications requiring fast and reliable image analysis.

## Contents

- [Usage](#usage)
- [Features](#features)
- [Installation](#installation)
- [Technologies Used](#technologies-used)
- [API](#api)
  - [Endpoints](#endpoints)
    - [/analyze](#analyze)
    - [/metadata](#metadata)
    - [/verify](#verify)
- [Contributing](#contributing)
- [License](#license)

## Features
- Analyze Photo: Detects faces and analyzes their race, gender, emotion, and age.
- Photo Metadata: Retrieves full metadata of a photo.
- Verify Identity: Compares faces in different photos to check if they belong to the same person.

## Usage

To use the FeelFlow, simply make HTTP requests to the specified endpoints, providing the necessary parameters as required by each endpoint. The API will respond with the relevant analysis results in JSON format.

## Installation

To get started with the Feel Flow, follow these steps:

1. Clone this repository to your local machine and open it.
```shell
git clone https://github.com/troubleShooter239/feel-flow-py.git
cd feel-flow-py
```
3. Install the required dependencies in your virtual environment.
```shell
python -m venv venv

source env/bin/activate
# or for windows
venv/scripts/activate

pip install -r requirements.txt
```
4. Run the application.
```shell
cd src
python main.py
```
5. Access the API endpoints as described below.

## Technologies Used

- **FastAPI**: High-performance web framework for building APIs.
- **TensorFlow**: Open-source platform for machine learning.
- **Keras**: Deep learning API written in Python.
- **NumPy**: Fundamental package for scientific computing in Python.
- **OpenCV**: Library for computer vision.
- **Pillow**: Python Imaging Library (PIL Fork) for opening, manipulating, and saving image files.
- **Pydantic**: Data validation and settings management using Python type annotations.

## API

### Endpoints

#### `/analyze`

This endpoint is used for analyzing photos and identifying human faces within them. If faces are detected, the API responds with a JSON object containing information about the detected faces.

**Example POST request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/analyze' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "b64_img": "data:image/jpeg;base64",
  "actions": {
    "age": true,
    "emotion": true,
    "gender": true,
    "race": true
  }
}'
```

It accepts a JSON payload containing base64-encoded image data along with optional actions to perform during analysis.

Here is an example response format:

**Response:**

```json
{
  "faces": [
    {
      "region": {
        "x": 0,
        "y": 0,
        "w": 0,
        "h": 0
      },
      "face_confidence": 0,
      "actions": {
        "age": 0,
        "emotion": {
          "angry": 0,
          "disgust": 0,
          "fear": 0,
          "happy": 0,
          "sad": 0,
          "surprise": 0,
          "neutral": 0
        },
        "dominant_emotion": "string",
        "gender": {
          "woman": 0,
          "man": 0
        },
        "dominant_gender": "string",
        "race": {
          "asian": 0,
          "indian": 0,
          "black": 0,
          "white": 0,
          "middle_eastern": 0,
          "latino_hispanic": 0
        },
        "dominant_race": "string"
      }
    }
  ]
}
```

#### `/metadata`

This endpoint is used for extracting metadata from images. It accepts a JSON payload containing base64-encoded image data.

**Example POST Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/metadata' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "b64_img": "Base64 Encoded String"
}'
```

**Response:**

```json
{
  "Summary": {
    "BBox": [
      null,
      null,
      null,
      null
    ],
    "BandNames": [
      "string"
    ],
    "DateTime": "string",
    "ExifOffset": 0,
    "Extrema": [
      null,
      null
    ],
    "FileType": "string",
    "FormatDescription": "string",
    "HasTransparency": true,
    "HostComputer": "string",
    "ImageSize": [
      null,
      null
    ],
    "Make": "string",
    "Megapixels": 0,
    "Mime": "string",
    "Mode": "string",
    "Model": "string",
    "Readonly": true,
    "Software": "string",
    "XResolution": 0,
    "YCbCrPositioning": 0,
    "YResolution": 0
  },
  "Exif": {
    "ApertureValue": 0,
    "BrightnessValue": 0,
    "ColorSpace": 0,
    "ComponentsConfiguration": "string",
    "CompositeImage": 0,
    "DateTimeDigitized": "string",
    "DateTimeOriginal": "string",
    "ExifImageHeight": 0,
    "ExifImageWidth": 0,
    "ExifVersion": "string",
    "ExposureBiasValue": 0,
    "ExposureMode": 0,
    "ExposureProgram": 0,
    "ExposureTime": 0,
    "FNumber": 0,
    "Flash": 0,
    "FlashPixVersion": "string",
    "FocalLength": 0,
    "FocalLengthIn35mmFilm": 0,
    "ISOSpeedRatings": 0,
    "LensMake": "string",
    "LensModel": "string",
    "LensSpecification": [
      null,
      null,
      null,
      null
    ],
    "MeteringMode": 0,
    "OffsetTime": "string",
    "OffsetTimeDigitized": "string",
    "OffsetTimeOriginal": "string",
    "SceneCaptureType": 0,
    "SceneType": "string",
    "SensingMethod": 0,
    "ShutterSpeedValue": 0,
    "SubjectLocation": [
      null,
      null,
      null,
      null
    ],
    "SubsecTimeDigitized": 0,
    "SubsecTimeOriginal": 0,
    "WhiteBalance": 0
  },
  "GPSInfo": {
    "GPSAltitude": 0,
    "GPSAltitudeRef": "string",
    "GPSDateStamp": "string",
    "GPSDestBearing": 0,
    "GPSDestBearingRef": "string",
    "GPSHPositioningError": 0,
    "GPSImgDirection": 0,
    "GPSImgDirectionRef": "string",
    "GPSLatitude": [
      null,
      null,
      null
    ],
    "GPSLatitudeRef": "string",
    "GPSLongitude": [
      null,
      null,
      null
    ],
    "GPSLongitudeRef": "string",
    "GPSSpeed": 0,
    "GPSSpeedRef": "string",
    "GPSTimeStamp": [
      null,
      null,
      null
    ]
  },
  "IFD1": {
    "Compression": 0,
    "JpegIFByteCount": 0,
    "JpegIFOffset": 0,
    "ResolutionUnit": 0,
    "XResolution": 0,
    "YResolution": 0
  }
}
```

#### `/verify`

This endpoint is used for face verification. It accepts a JSON payload containing base64-encoded image data for two images, along with other parameters.

**Example POST Request:**

```bash
curl -X 'POST' \
  'http://localhost:8000/verify' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "b64_img1": "data:image/jpeg;base64",
  "b64_img2": "data:image/jpeg;base64",
  "r_model_name": "vgg-face",
  "distance_metric": "cosine"
}'
```

**Response:**

```json
{
  "verified": true,
  "distance": 0,
  "threshold": 0,
  "model": "string",
  "distance_metric": "string",
  "facial_areas": {
    "img1": {
      "x": 0,
      "y": 0,
      "w": 0,
      "h": 0
    },
    "img2": {
      "x": 0,
      "y": 0,
      "w": 0,
      "h": 0
    }
  }
}
```

Replace `"b64_img1"`,  `"b64_img2"`, `"r_model_name"`, and `"distance_metric"` with the actual image`s data, model name, and distance metric you want to use for verification.

## Contributing

Contributions to the Feel Flow are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
