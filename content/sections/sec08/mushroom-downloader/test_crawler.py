import sys
import os
import requests
import time
import shutil
from selenium import webdriver

import threading
from queue import Queue

thread_count = 50
threads = []
# Initial queue
queue = Queue(0)

class Downloader(threading.Thread):
    def __init__(self, queue, thread):
        threading.Thread.__init__(self)
        self.queue = queue
        self.thread = thread

    def run(self):
        while self.queue.empty() == False:
            item = self.queue.get()

            #print("Thread:",self.thread,item)
            #time.sleep(3)
            download_from_url(item["url"], item["img_dir"], item["file_path"])

            self.queue.task_done()

def download_from_url(url,img_dir,file_path):
    try:
        img_path = os.path.basename(url)
        #file_path = os.path.join(img_dir, img_path)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print("Error in url:", url)
        print(e)

def download_google_images(search_term_list = ["crimini mushroom","oyster mushroom","amanita mushroom"], num_images_requested=5):
    print("download_google_images...")
    start_time = time.time()

    # Setup download folder
    downloads = "dataset"
    if os.path.exists(downloads):
        shutil.rmtree(downloads)
    os.mkdir(downloads)

    # Each scrolls provides 400 image approximately
    number_of_scrolls = int(num_images_requested / 400) + 1

    # Chrome Options
    options = webdriver.ChromeOptions()
    #options.headless = True
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')

    # Create a browser instance
    browser = webdriver.Chrome("driver/chromedriver", chrome_options=options)

    for search_term in search_term_list:
        print("Searching for :", search_term)
        browser.get('https://www.google.com/search?q=' + search_term)

        # Go to Google Images
        images_links = browser.find_elements_by_xpath('//a[contains(@class, "hide-focus-ring")]')
        for link in images_links:
            #print(link)
            print(link.get_attribute("href"))

        # Wait
        time.sleep(1)

        # Go to images
        images_link = images_links[0]
        images_link.click()

        for _ in range(number_of_scrolls):
            for __ in range(10):
                # multiple scrolls needed to show all 400 images
                browser.execute_script("window.scrollBy(0, 1000000)")
                time.sleep(1)
            # to load next 400 images
            time.sleep(2)
            # try to find show more results bottom
            try:
                # if found click to load more image
                browser.find_element_by_xpath("//input[@value='Show more results']").click()
            except Exception as e:
                # if not exit
                print("End of page")
                break

        # Image links
        imgs_urls = set()
        # Find the thumbnail images
        thumbnails = browser.find_elements_by_xpath('//a[@class="wXeWr islib nfEiy mM5pbd"]')
        # loop over the thumbnails to retrive the links
        for thumbnail in thumbnails:
            # check if we reached the requested number of links
            if len(imgs_urls) >= num_images_requested:
                break
            try:
                thumbnail.click()
                time.sleep(1)
            except Exception as error:
                print("Error clicking one thumbnail : ", error)
            # Find the image url
            url_elements = browser.find_elements_by_xpath('//img[@class="n3VNCb"]')
            # check for the correct url
            for url_element in url_elements:
                try:
                    url = url_element.get_attribute('src')
                except e:
                    print("Error getting url")
                if url.startswith('http') and not url.startswith('https://encrypted-tbn0.gstatic.com'):
                    #print("Found image url:", url)
                    imgs_urls.add(url)

        print('Number of image urls found:', len(imgs_urls))

        # Wait 5 seconds
        time.sleep(2)

        # Save the images
        img_dir = os.path.join(downloads, search_term.lower().replace(" ","_"))
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)

        count = 0
        for url in imgs_urls:
            file_path = os.path.join(img_dir, '{0}.jpg'.format(count))
            count += 1
            queue.put({"url": url, "img_dir": img_dir,"file_path":file_path})

        # Execute downloads from queue in a thread
        for i in range(thread_count):
            thread = Downloader(queue, i)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    # Quit the browser
    browser.quit()

    execution_time = (time.time() - start_time) / 60.0
    print("Download execution time (mins)", execution_time)


# Download images
download_google_images(search_term_list = ["oyster mushroom","amanita mushroom"], num_images_requested=10)

