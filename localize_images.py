from PIL import Image
import requests
from io import BytesIO

link_list = []

# Takes a URL and sucks the image out of it
# I'm keeping that phrasing
def skim_image(url, num):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    path_name = 'C:/Users/Cassandra/C Coding/CNN/Training_Images/' + str(num) + '.jpg'
    img.save(path_name, 'JPEG')