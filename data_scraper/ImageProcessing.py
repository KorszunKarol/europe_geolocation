from dataclasses import dataclass, field
import cv2
from typing import List
from geopy.geocoders import Nominatim
import os
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import pickle
import csv
import pandas as pd
import folium

@dataclass
class ImageProcessing:
    images: List[cv2.Mat]
    coords: List[float]
    stitched_image: cv2.Mat = field(init=False, default=None)
    folder_path: str

    def __post_init__(self):
        self.stitched_image = self.preprocess_images()


    def image_to_panorama(self, im1, im2, im3, im4):
        images = [im1, im2, im3, im4]
        rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

        height = 480
        width = 2560 // 4
        resized_images = [cv2.resize(img, (width, height)) for img in rgb_images]

        # Concatenate the images horizontally
        collage = np.hstack(resized_images)
        return collage

    def preprocess_images(self):
        print(f"Number of images: {len(self.images)}")
        for i, img in enumerate(self.images[:4]):
            print(f"Image {i} shape: {img.shape}")
            print(f"Image {i} dtype: {img.dtype}")

        # Check if we have at least 4 images
        if len(self.images) < 4:
            print("Error: Not enough images for panorama")
            return None

        # Create panorama using the new method
        panorama = self.image_to_panorama(*self.images[:4])

        if panorama is not None:
            print("Panorama created successfully")
            return panorama
        else:
            print("Failed to create panorama")
            return None

    def stitch_original(self):
        return self.stitch_images(self.images[:4])

    def stitch_grayscale(self):
        gray_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.images[:4]]
        return self.stitch_images(gray_images)

    def stitch_resized(self):
        resized_images = [cv2.resize(img, (0,0), fx=0.5, fy=0.5) for img in self.images[:4]]
        return self.stitch_images(resized_images)

    def stitch_pairs(self):
        stitcher = cv2.Stitcher_create()
        status1, stitched1 = stitcher.stitch([self.images[0], self.images[1]])
        status2, stitched2 = stitcher.stitch([self.images[2], self.images[3]])

        if status1 == cv2.Stitcher_OK and status2 == cv2.Stitcher_OK:
            return self.stitch_images([stitched1, stitched2])

        print("Pair stitching failed")
        return None

    def stitch_images(self, images_to_stitch):
        try:
            stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            stitcher.setRegistrationResol(0.6)
            stitcher.setSeamEstimationResol(0.1)
            stitcher.setCompositingResol(1)
            stitcher.setPanoConfidenceThresh(1)

            status, stitched = stitcher.stitch(images_to_stitch)

            if status == cv2.Stitcher_OK:
                return stitched
            else:
                error_messages = {
                    cv2.Stitcher_ERR_NEED_MORE_IMGS: "Not enough images",
                    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Homography estimation failed",
                    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Camera parameter adjustment failed"
                }
                error_msg = error_messages.get(status, f"Unknown error (status: {status})")
                print(f"Image stitching failed: {error_msg}")
                return None
        except Exception as e:
            print(f"An error occurred during stitching: {str(e)}")
            return None

    def get_image_dimensions(self):
        if self.stitched_image is not None:
            height, width = self.stitched_image.shape[:2]
            return height, width
        else:
            return 0, 0

    def get_dataset_length(self):
        supported_formats = (".jpg", ".jpeg", ".png")
        image_files = [
            f for f in os.listdir(self.folder_path) if f.endswith(supported_formats)
        ]
        return len(image_files)

    def save_image(self):
        ds_len = self.get_dataset_length()
        lat = np.float16(self.coords[0])
        lon = np.float16(self.coords[1])
        filename = f"{ds_len}_{lat}_{lon}.jpg"

        if self.stitched_image is not None:
            full_path = os.path.join(self.folder_path, filename)
            cv2.imwrite(full_path, cv2.cvtColor(self.stitched_image, cv2.COLOR_RGB2BGR))
            self.save_coords_to_csv(ds_len, lat, lon, "coords.csv")
            print(f"Panorama saved as {full_path}.")
        else:
            print("No panorama to save.")

    def get_country_from_coords(self):
        try:
            base_url = "https://nominatim.openstreetmap.org/reverse"
            lat, lon = self.coords
            print(lat)
            print(lon)
            params = {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}

            retry_strategy = Retry(
                total=3,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"],
                backoff_factor=1,
            )

            adapter = HTTPAdapter(max_retries=retry_strategy)
            session = requests.Session()
            session.mount("https://", adapter)

            response = session.get(base_url, params=params, timeout=5)
            response.raise_for_status()

            location = response.json()
            address = location["address"]
            country = address.get("country", "Country not found")
            return country.replace(" ", "")
        except Exception as e:
            print(f"Error retrieving country: {e}")
            return None


    def save_coords_to_csv(self, idx, lat, lon, csv_path):
        header = ['ID', 'Latitude', 'Longitude']
        file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            writer.writerow([idx, lat, lon])

def plot_scatter_locations_from_csv(csv_file):
    # Read the CSV file
    data = pd.read_csv(csv_file)

    # Check if the CSV has 'latitude' and 'longitude' columns
    if 'Latitude' not in data.columns or 'Longitude' not in data.columns:
        print("CSV file must contain 'latitude' and 'longitude' columns")
        return

    # Create a map centered around the mean location
    map_center = [data['Latitude'].mean(), data['Longitude'].mean()]
    map = folium.Map(location=map_center, zoom_start=5)

    # Plot each location as a circle (scatter plot)
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=3, # Size of the circle marker
            color='blue',
            fill=True,
            fill_color='blue'
        ).add_to(map)

    # Save the map to an HTML file
    map.save('scatter_map.html')
    print("Scatter plot map has been saved to 'scatter_map.html'. Open this file in a web browser to view the map.")


def main():
    # folder_path = "images/"
    # coords = [10.476924896240234, 2.1142188087105751]

    # imgs_file_path = "images_imgs.pkl"

    # with open(imgs_file_path, "rb") as f:
    #     images = pickle.load(f)

    # image_processor = ImageProcessing(
    #     images=images, coords=coords, folder_path=folder_path
    # )

    # if image_processor.stitched_image is not None:
    #     image_processor.save_image()
    # else:
    #     print("Stitching failed, no image to save.")

    # country = image_processor.get_country_from_coords()
    # print(f"Country retrieved: {country}")
    plot_scatter_locations_from_csv("coords.csv")



if __name__ == "__main__":
    main()