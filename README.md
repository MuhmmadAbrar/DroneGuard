# Drone Map Generator and Image Segmentation

This project is a Streamlit application that allows users to stitch drone images to generate maps and perform image segmentation to identify water bodies.

## Features

- Map stitching from multiple drone images
- Water body segmentation using a custom-trained U-Net model
- Visualization of segmented water bodies on the stitched map
- Display of water body coordinates.

## Prerequisites

- Python 3.6 or higher
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
   `git clone https://github.com/your-username/drone-map-generator.git`

2. Navigate to the project directory:
   `cd drone-map-generator`

3. Install the required Python packages:
   `pip install -r requirements.txt`

## Usage

1. Run the Streamlit app:
   `streamlit run app.py`

2. Upload drone images for map stitching.
3. Click the "Generate Map" button to stitch the images and perform water body segmentation.
4. View the stitched map, segmented water bodies, and water content information.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push your changes to your fork
5. Submit a pull request

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the user interface
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch) for the U-Net model
- [OpenCV](https://opencv.org/) for image processing
