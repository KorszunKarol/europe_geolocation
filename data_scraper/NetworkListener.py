from dataclasses import dataclass, field
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import json
import pprint


@dataclass
class NetworkTrafficListener:
    log_file: str = "log_entries.txt"
    driver: webdriver.Chrome = field(default=None, init=True)

    def process_browser_logs_for_network_events(self, logs):
        for entry in logs:
            print(entry)
            # log = json.loads(entry["message"])["message"]
            # if any(
            #     event in log["method"]
            #     for event in [
            #         "Network.response",
            #         "Network.request",
            #         "Network.webSocket",
            #     ]
            # ):
            #     yield log

    def capture_and_process_logs(self):
        logs = self.driver.get_log("server")
        events = self.process_browser_logs_for_network_events(logs)
        with open(self.log_file, "wt") as out:
            for event in events:
                pprint.pprint(event, stream=out)

    def close_driver(self):
        if self.driver:
            self.driver.quit()
