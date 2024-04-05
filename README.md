# FeelFlow-API

This is the official repository for the FeelFlow API, a service designed to analyze media content using neural processing techniques. The API is built on FastAPI and UVicorn, providing endpoints for various functionalities related to media analysis, particularly focusing on detecting human faces within images.

## Contents

- [Usage](#usage)
- [Getting Started](#getting-started)
- [API](#api)
  - [Endpoints](#endpoints)
    - [/analyze](#analyze)
    - [/metadata](#metadata)
    - [/verify](#verify)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Usage

To use the FeelFlow API, simply make HTTP requests to the specified endpoints, providing the necessary parameters as required by each endpoint. The API will respond with the relevant analysis results in JSON format.

## Getting Started

To get started with the FeelFlow API, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the application using `uvicorn main:app --reload`.
4. Access the API endpoints as described below.

## API

### Endpoints

#### `/analyze`

This endpoint is used for analyzing photos and identifying human faces within them. If faces are detected, the API responds with a JSON object containing information about the detected faces.

**Example POST request:**

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "b64_img": "base64_encoded_image_data",
    "actions": {
      "age": true,
      "emotion": true,
      "gender": true,
      "race": true
    }
  }'
```

It accepts a JSON payload containing base64-encoded image data along with optional actions to perform during analysis.

**Request JSON Schema:**

```json
{
  "b64_img": "base64_encoded_image_data",
  "actions": {
    "age": true,
    "emotion": true,
    "gender": true,
    "race": true
  }
}
```

Here is an example response format:

**Response:**

```json
[
  {
    "region": {"x": 837, "y": 856, "w": 1813, "h": 1813},
    "face_confidence": 5.754483174474444,
    "age": 32,
    "emotion": {
      "angry": 5.26,
      "disgust": 0.9,
      "fear": 39.82,
      "happy": 1.79,
      "sad": 44.55,
      "surprise": 2.06,
      "neutral": 5.62
    },
    "dominant_emotion": "sad",
    "gender": {
      "woman": 15.96,
      "man": 84.04
    },
    "dominant_gender": "man",
    "race": {
      "asian": 0.16,
      "indian": 0.16,
      "black": 0.01,
      "white": 83.35,
      "middle_eastern": 8.07,
      "latino_hispanic": 8.24
    },
    "dominant_race": "white"
  }
]
```

#### `/metadata`

This endpoint is used for extracting metadata from images. It accepts a JSON payload containing base64-encoded image data.

**Example POST Request:**

```bash
curl -X POST "http://localhost:8000/metadata" \
  -H "Content-Type: application/json" \
  -d '{
    "b64_img": "base64_encoded_image_data"
  }'
```

**Request JSON Schema:**

```json
{
  "b64_img": "base64_encoded_image_data"
}
```

#### `/verify`

This endpoint is used for face verification. It accepts a JSON payload containing base64-encoded image data for two images, along with other parameters.

**Example POST Request:**

```bash
curl -X POST "http://localhost:8000/verify" \
  -H "Content-Type: application/json" \
  -d '{
    "b64_img1": "base64_encoded_image_data_1",
    "b64_img2": "base64_encoded_image_data_2",
    "r_model_name": "model_name",
    "distance_metric": "metric"
  }'
```

**Request JSON Schema:**

```json
{
  "b64_img1": "base64_encoded_image_data_1",
  "b64_img2": "base64_encoded_image_data_2",
  "r_model_name": "model_name",
  "distance_metric": "metric"
}
```

Replace `"base64_encoded_image_data"`, `"model_name"`, and `"metric"` with the actual image data, model name, and distance metric you want to use for verification.

## Technologies Used

- **FastAPI**: FastAPI is used as the web framework for building APIs with Python 3.7+ based on standard Python type hints.
- **UVicorn**: UVicorn is used as the ASGI server to serve the FastAPI application.
- **Neural Processing**: Neural processing techniques are employed for analyzing media content, particularly in detecting human faces and extracting related attributes such as age, gender, emotions, and race.

## Contributing

Contributions to the FeelFlow API are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.
