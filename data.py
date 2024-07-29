import requests
from PIL import Image
import io
import os
import random
from global_land_mask import globe
from stitch_images import image_to_panorama
import numpy as np
import cv2

def get_street_view_image(api_key, location, size=(640, 480), heading=0, radius=100000, return_error_code=True):

    base_url = "https://maps.googleapis.com/maps/api/streetview"
    base_metadata_url = "https://maps.googleapis.com/maps/api/streetview/metadata"
    url = f"{base_url}?size={size[0]}x{size[1]}&location={location[0]}, {location[1]}&key={api_key}&heading={heading}&radius={radius}&return_error_code={return_error_code}"
    metadata_url = f"{base_metadata_url}?size={size[0]}x{size[1]}&location={location[0]}, {location[1]}&key={api_key}&heading={heading}&radius={radius}&return_error_code={return_error_code}"

    metadata_response = requests.get(metadata_url)
    official_coverage = is_official_coverage(metadata_response.json())

    if official_coverage:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        return image
    else:
        print("Error: Unable to fetch street view image.")
        return None

def get_random_location(lat_lower=-90, lat_upper=90, lon_lower=-90, lon_upper=90):
    latitude = random.uniform(lat_lower, lat_upper)
    longitude = random.uniform(lon_lower, lon_upper)
    return latitude, longitude

def get_continent_data(num_coords, lat_lower, lat_upper, lon_lower, lon_upper):
    coords = []
    while len(coords) < num_coords:
        lat, lon = get_random_location(lat_lower=lat_lower, lat_upper=lat_upper, lon_lower=lon_lower, lon_upper=lon_upper)
        if globe.is_land(lat, lon):
            coords.append((lat, lon))

    write_coordinates_to_txt("europe_2.txt", coords)


def download_images(file_path, api_key, size, radius):
    coords = read_coordinates_from_txt(file_path)
    for i, location in enumerate(coords):
        image = get_street_view_image(api_key, location, size=size, heading=0, radius=radius)
        if image is not None:
            image_1 = np.array(image)
            image_east = np.array(get_street_view_image(api_key, location, size=size, heading=90, radius=radius))
            image_south = np.array(get_street_view_image(api_key, location, size=size, heading=180, radius=radius))
            image_west = np.array(get_street_view_image(api_key, location, size=size, heading=270, radius=radius))
            image = image_to_panorama(image_1, image_east, image_south, image_west)
            cv2.imwrite(f"street_view_images_europe_2/street_view_{i + 19_422}.jpg", image)
            print(f"Image {i + 1} saved.")

def write_coordinates_to_txt(filename, coordinates):
    with open(filename, "a") as file:
        for lat, lon in coordinates:
            file.write(f"{lat},{lon}\n")

def read_coordinates_from_txt(filename):
    coordinates = []
    with open(filename, "r") as file:
        for line in file:
            lat, lon = map(float, line.strip().split(","))
            coordinates.append((lat, lon))
    return coordinates

def is_official_coverage(metadata):
    copyright_notice = metadata.get("copyright", "")
    copyright_notice_lower = copyright_notice.lower()

    if "google" in copyright_notice_lower:
        return True
    else:
        return False
