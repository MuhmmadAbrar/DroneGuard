# Drone Map Stitcher and Image Segmentation

This Streamlit application empowers users with vital capabilities for flood monitoring in disaster zones, especially when communication channels are disrupted by natural disasters. It combines drone image stitching to generate comprehensive maps with advanced image segmentation techniques. Future developments aim to seamlessly integrate this functionality into autonomous drone systems, enabling faster and more streamlined disaster response.

## Features

- Map stitching from multiple drone images.
- Water body segmentation using a custom-trained U-Net model.
- Visualization of segmented water bodies on the stitched map.
- Display of water body coordinates.

## Prerequisites

- Python 3.6 or higher
- Required Python packages listed in `requirements.txt`.

## Installation

1. Clone the repository:
   `git clone https://github.com/MuhmmadAbrar/DroneGuard.git`

2. Navigate to the project directory:
   `cd DroneGuard`

3. Install the required Python packages:
   `pip install -r requirements.txt`

## Usage

1. Run the Streamlit app:
   `streamlit run app.py`

2. Upload drone images for map stitching.
3. Click the "Generate Map" button to stitch the images and perform water body segmentation.
4. View the stitched map, segmented water bodies, and water content information.

## Results:

1. **Home Page**

   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/ed5d39c3-2eb0-40ae-9999-728f41b81ae3" width="500">

2. **Map Stitching**

   1. **Uploaded Images:**

   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/08450f90-a22b-4b49-bfb0-13a3aa4f51db" width="400">
   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/86cd3ec0-170d-49a5-b60b-cfa1b63027b8" width="400">
   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/c7698853-c805-4ee2-82f3-2e72deb1bbbb" width="400">
   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/5c78c417-145e-463b-ac1d-6062ff176d4d" width="400">

   2. **Result:**

   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/b793ab7d-6b08-43c5-95d1-3a7eac29d846" width="800">

3. **Image Segmentation**

   1. **Uploaded Image:**

   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/6fbd9291-a31d-454d-8d2e-4988b880560b" width="500">

   2. **Result:**

   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/d3e1b968-7af8-4d61-afe2-73e6840dbc60" width="500">
   - <img src="https://github.com/MuhmmadAbrar/DroneGuard/assets/88892675/d68e0756-ce98-4fc7-925d-67ea8463a1d7" width="500">

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
