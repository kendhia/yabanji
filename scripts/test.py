import scrapy 

class BlogSpider(scrapy.Spider):
    name = 'blogspider'
    start_urls = ['https://twitter.com/kendhia']

    def parse(self, response):
        for title in response.css('.content'):
            yield {'title': title}

        for next_page in response.css('a.next-posts-link'):
            yield response.follow(next_page, self.parse)