from dataclasses import dataclass, field
from seleniumwire import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import os
import time
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from ImageProcessing import ImageProcessing
import requests
import json
import re
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import unquote
import numpy as np
import cv2
import pickle
import brotli


@dataclass
class ScraperConfig:
    email: str = os.environ.get("GEOGUESSR_EMAIL")
    password: str = os.environ.get("GEOGUESSR_PASSWORD")
    map_url: str = "https://www.geoguessr.com/maps/60788ebfb15d750001929514"


@dataclass
class DataScraper:
    config: ScraperConfig
    driver: webdriver.Chrome = field(init=False, default=None)
    token: str = field(init=False, default=None)
    coords: list = field(init=False, default=None)

    def __post_init__(self):
        options = Options()
        user_data_dir = r"C:\Users\korsz\AppData\Local\Google\Chrome\User Data"
        profile_directory = "Profile 1"
        options.add_argument(f"user-data-dir={user_data_dir}")
        options.add_argument(f"profile-directory={profile_directory}")

        # options.add_argument("--headless")  # Enable headless mode
        options.add_argument("--disable-gpu")  # Recommended when running headless
        options.add_argument(
            "--no-sandbox"
        )  # Bypass OS security model, required on Linux if running as root
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )
        self.session = requests.Session()
        retries = Retry(
            total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def login(self):
        self.driver.get(self.config.map_url)

        # login_button = WebDriverWait(self.driver, 10).until(
        #     EC.element_to_be_clickable(
        #         (By.CSS_SELECTOR, "a[data-qa='header-login-button']")
        #     )
        # )

        # login_button.click()

        # email_input = WebDriverWait(self.driver, 10).until(
        #     EC.visibility_of_element_located((By.NAME, "email"))
        # )
        # email_input.send_keys(os.environ.get("GEOGUESSR_EMAIL"))

        # password_input = WebDriverWait(self.driver, 10).until(
        #     EC.visibility_of_element_located((By.NAME, "password"))
        # )
        # password_input.send_keys(os.environ.get("GEOGUESSR_PASSWORD"))
        # submit_button = WebDriverWait(self.driver, 10).until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
        # )
        # submit_button.click()

    def play_round(self, base_filename="images"):
        action = ActionChains(self.driver)
        imgs = []
        self.driver.find_element(By.TAG_NAME, "body").click()
        for i in range(1, 5):
            time.sleep(0.4)
            screenshot_png = self.driver.get_screenshot_as_png()
            nparr = np.frombuffer(screenshot_png, np.uint8)
            img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            imgs.append(img_cv)
            action.key_down(Keys.ARROW_RIGHT).perform()
            time.sleep(0.95)
            action.key_up(Keys.ARROW_RIGHT).perform()
            # cv2.imshow(f"Image {i}", img_cv)
            # cv2.waitKey(4)
        self.make_guess()
        ip = ImageProcessing(images=imgs, coords=self.coords, folder_path="images/")
        ip.save_image()

        cv2.destroyAllWindows()

    def play_game(self):
        play_button = WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    ".button_variantPrimary__u3WzI.button_sizeLarge__nKm9V",
                )
            )
        )
        play_button.click()

        # toggle_button = WebDriverWait(self.driver, 10).until(
        #     EC.element_to_be_clickable((By.CSS_SELECTOR, ".toggle_toggle__qfXpL"))
        # )
        # toggle_button.click()

        # slider = WebDriverWait(self.driver, 10).until(
        #     EC.element_to_be_clickable(
        #         (
        #             By.CSS_SELECTOR,
        #             "div.game-options_optionInput__paPBZ input.toggle_toggle__qfXpL",
        #         )
        #     )
        # )
        # slider.click()

        start_game_button = WebDriverWait(self.driver, 20).until(
            EC.element_to_be_clickable(
                (By.CSS_SELECTOR, "button[data-qa='start-game-button']")
            )
        )
        start_game_button.click()
        while not self.token:
            req = self.driver.requests
            self.retrieve_token(req)

    def dump_response_to_file(self, response_body, filename="raw_response.txt"):
        directory = "response_dumps"
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)

        with open(file_path, "wb") as f:
            f.write(response_body)

        print(f"Raw response dumped to {file_path}")

    def retrieve_token(self, requests):

        with open("requests_dump.txt", "w") as file:
            for request in requests:
                file.write(str(request) + "\n")
        token = None
        for request in requests:
            request_url = (
                request
                if isinstance(request, str)
                else request.url if hasattr(request, "url") else None
            )
            if request_url:
                decoded_url = unquote(request_url)
                match = re.search(r"/game/([^&.]+)", decoded_url)
                if match:
                    token = match.group(1)
                    print(f"token: {token}")
                    self.token = token
                    return
        if token:
            self.token = token

    def get_model_guess(self):
        url = f"https://www.geoguessr.com/api/v3/games/{self.token}"
        response = self.session.get(url)
        response.raise_for_status()
        response_data = response.json()
        self.parse_response(response_data)

    def make_guess(self):
        url = f"https://www.geoguessr.com/api/v3/games/{self.token}"

        payload = {"token": self.token, "lat": 49.000, "lng": 15.000, "timedOut": False}

        browser_cookies = self.driver.get_cookies()

        for cookie in browser_cookies:
            self.session.cookies.set(
                cookie["name"], cookie["value"], domain=cookie["domain"]
            )

        headers = {
            "authority": "www.geoguessr.com",
            "accept": "*/*",
            "accept-language": "en-US,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://www.geoguessr.com",
            "referer": f"https://www.geoguessr.com/game/{self.token}",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "x-client": "web",
        }

        try:
            response = self.session.post(url, json=payload, headers=headers)
            response.raise_for_status()
            self.parse_response(response_data=response.json())

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")

    def start_new_game(self):
        # Define the API endpoint URL
        url = "https://www.geoguessr.com/api/v3/games"  # Adjust this URL to the correct endpoint

        # Define the payload with the game settings
        payload = {
            "map": "62a44b22040f04bd36e8a914",
            "type": "standard",
            "timeLimit": 0,
            "forbidMoving": True,
            "forbidRotating": False,
            "forbidZooming": False,
            "rounds": 5,
        }

        headers = {
            ":authority": "www.geoguessr.com",
            ":method": "POST",
            ":path": "/api/v3/games",
            ":scheme": "https",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br, zstd",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Client": "web",
        }

        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            game_data = response.json()
            print("Game started successfully:", game_data)
            # Extract and return the game token or ID if needed
            return game_data
        else:
            # Handle errors
            print("Failed to start game:", response.text)
            return None

    def parse_response(self, response_data):
        coords_list = response_data["rounds"]
        round_num = len(coords_list)
        self.coords = [
            coords_list[round_num - 1]["lat"],
            coords_list[round_num - 1]["lng"],
        ]
        print(self.coords)

    def parse_new_game_response(self):
        request = self.driver.wait_for_request("/api/v3/games", timeout=30)
        if request and request.response:
            print("Request headers:", request.headers)
            print("Response headers:", request.response.headers)

            response_body = request.response.body
            self.dump_response_to_file(response_body)

            decoded_data = self.decode_response("response_dumps/raw_response.txt")
            if decoded_data and "token" in decoded_data:
                self.token = decoded_data["token"]
                print(f"New game token: {self.token}")
            return decoded_data
        else:
            print("No response captured for /api/v3/games")
        return None

    def decode_response(self, file_path):
        with open(file_path, "rb") as f:
            response_body = f.read()

        try:
            # Decompress using Brotli
            decompressed_body = brotli.decompress(response_body)

            # Decode to string
            decoded_body = decompressed_body.decode("utf-8")

            # Parse JSON
            response_data = json.loads(decoded_body)

            return response_data
        except Exception as e:
            print(f"Failed to decompress or parse: {str(e)}")
        return None

    def run(self):
        self.login()
        self.play_game()
        while True:
            try:
                for i in range(5):
                    WebDriverWait(self.driver, 40).until(
                        EC.visibility_of_element_located(
                            (
                                By.CSS_SELECTOR,
                                "div[data-qa='map-name'] .status_value__w_Nh0",
                            )
                        )
                    )
                    # self.hide_elements()
                    self.play_round()
                    self.driver.refresh()

                del self.driver.requests

                actions = ActionChains(self.driver)
                actions.send_keys(Keys.SPACE).perform()

                actions.send_keys(Keys.SPACE).perform()
                self.parse_new_game_response()
            except Exception as e:
                self.driver.refresh()
                print(f"Error: {e}")
                pass

    def hide_elements(self):
        try:
            self.driver.execute_script(
                """
                // Existing code to hide elements
                var element = document.querySelector('.status_inner__eAJp4');
                if (element) {
                    element.style.display = 'none';
                }

                var spanElement = document.getElementById('EBEFD861-FAC5-4D1E-9760-E2FC39427AE1');
                if (spanElement) {
                    spanElement.style.display = 'none';
                }

                var buttonElement = document.querySelector('button[data-qa="perform-guess"]');
                if (buttonElement) {
                    buttonElement.style.display = 'none';
                }

                var geoGuessrLogo = document.querySelector('img[alt="GeoGuessr"]');
                if (geoGuessrLogo) {
                    geoGuessrLogo.style.display = 'none';
                }

                var latitudeLabel = document.getElementById('latitude-label');
                if (latitudeLabel) {
                    latitudeLabel.style.display = 'none';
                }

                """
            )
            print("Elements hidden successfully")
        except Exception as e:
            print(f"Error hiding elements: {e}")


def main():
    config = ScraperConfig()
    scraper = DataScraper(config)
    res = scraper.run()
    print(res)


if __name__ == "__main__":
    main()
