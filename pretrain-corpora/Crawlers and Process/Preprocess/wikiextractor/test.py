import wget
import json
import pandas as pd
pd.set_option("max_colwidth", 600)
import ast
from bs4 import BeautifulSoup
import re
import requests
import time
import numpy as np
import zipfile
import os
import html
import re
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=False)


def download_bz2(url):
    filename = url.split('/')[-1]
    path = '/data/yibo/wiki/data/%s' % filename
    print(filename)

    if os.path.exists(path):
        pass
    else:
        print('download...', filename)
        real_url = 'https://dumps.wikimedia.org' + url
        filename = wget.download(real_url, out=path)

base_url = 'https://dumps.wikimedia.org/enwiki/20210320/'
coverpage = requests.get(base_url).content
soup = BeautifulSoup(coverpage, 'html.parser')
links = soup.find_all('a', text =\
                      re.compile('enwiki-20210320-pages-articles-multistream\d+.xml.*.bz2'), href=True)
df = pd.json_normalize([{'name': i.text,'url': i['href']} for i in links])
df = df[~df.duplicated(subset='url')]

from wikiextractor.main import main

for name in df.name:
    print('------  %s  -------'%name)
    main(input = 'data/%s'%name, output = 'output/%s'%name)