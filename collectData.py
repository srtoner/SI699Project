import pandas as pd
import numpy as np
import requests
import json

BASE_URL = "https://www.gutenberg.org/"
TOP_100 = BASE_URL + "browse/scores/top"

def download_book(self,booktitle,data_link):
    try:
        r=self.session.get(data_link)
        filename=self.data_directory+booktitle+".txt"

        file = open(filename, "w")
        file.write(r.text)
        file.close()
    except Exception as e:
        print(e)
        print("<--- ERROR DOWNLOADING %s --->" % (booktitle))





if __name__ == "__main__":

    

    pass