import luigi
import requests
from bs4 import BeautifulSoup
import json
import re
import os


WH_BASE_URL = 'https://obamawhitehouse.archives.gov'
WEEKLY_ADDRESS_URL = WH_BASE_URL + '/briefing-room/weekly-address?page='
WH_PAGE_COUNT = 44


class ScrapeWhiteHouse(luigi.Task):
    data_dir = luigi.Parameter('./data/')

    def output(self):
        return luigi.LocalTarget(os.path.join(self.data_dir, 'yt_video_ids.json'))

    def run(self):
        scraped_wh_urls = set()
        yt_video_ids = set()

        for i in range(WH_PAGE_COUNT):
            data = requests.get(WEEKLY_ADDRESS_URL + str(i)).content

            root = BeautifulSoup(data, 'html.parser').find_all(
                "div", class_="views-field views-field-title")
            for x in root:
                url_element = x.find('a')
                if url_element.text.lower().startswith('weekly address:'):
                    url = WH_BASE_URL + url_element['href']
                    if url not in scraped_wh_urls:
                        page_content = str(requests.get(url).content)

                        yt_video_match = re.search(
                            r'\<iframe.*?\/embed\/(.{11}).*?\<\/iframe\>', page_content)
                        if yt_video_match:
                            yt_video_ids.add(yt_video_match.group(1))

                        scraped_wh_urls.add(url)
            self.set_progress_percentage(round(i/WH_PAGE_COUNT)*100)

        with self.output().open('w') as f:
            json.dump(list(yt_video_ids), f)
