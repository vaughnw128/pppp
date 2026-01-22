# pppp (Penultimate Picture Post-Processor)

Image post processor for grabbing tags and OCR data from any image. Meant to provide metadata to \<??\> for better content understanding.

## Features

- Extracts metadata tags from images using machine learning models ([RAM++](https://github.com/xinyu1205/recognize-anything)).
- Performs OCR ([PaddleOCR](github.com/PaddlePaddle/PaddleOCR)) to extract text from images.
- Supports both raw byte inputs and image URLs.
- Is awesome

## TODO:

- VIT for sentiment analysis on images
- Support for video files
- Quieter logging
- Authentication for API access