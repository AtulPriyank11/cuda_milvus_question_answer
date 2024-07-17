import scrapy
from scrapy.crawler import CrawlerProcess
from urllib.parse import urlparse
from bs4 import BeautifulSoup

class NvidiaDocsSpider(scrapy.Spider):
    name = "nvidia_docs"
    start_urls = ['https://docs.nvidia.com/cuda/']

    custom_settings = {
        'DEPTH_LIMIT': 5,  # Limit the depth to 5 levels
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',  # Set log level to INFO to reduce verbosity
    }

    def parse(self, response):
        # Check if response content is text
        content_type = response.headers.get('Content-Type', b'').decode('utf-8')
        if 'text' in content_type:
            # Extract text content using BeautifulSoup
            soup = BeautifulSoup(response.body, 'html.parser')
            page_content = ' '.join(soup.stripped_strings)
            yield {
                'url': response.url,
                'content': page_content,
            }

            # Follow links
            for href in response.css('a::attr(href)').getall():
                href = response.urljoin(href)
                parsed_url = urlparse(href)
                if parsed_url.netloc == "docs.nvidia.com" and parsed_url.path.startswith('/cuda/') and not parsed_url.path.endswith(('.pdf', '.jpg', '.jpeg', '.png', '.zip', '.svg', '.gz', '.tar', '.exe')):
                    yield response.follow(href, self.parse)
        else:
            self.logger.warning(f"Skipped non-text content at {response.url}")

if __name__ == "__main__":
    process = CrawlerProcess(settings={
        'FEEDS': {
            'nvidia_docs.json': {'format': 'json'},  # Save data to JSON file
        },
    })

    process.crawl(NvidiaDocsSpider)
    process.start()
