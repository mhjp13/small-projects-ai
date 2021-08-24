import requests as r
from bs4 import BeautifulSoup

def main(term):
    base_url = "https://en.wikipedia.org/wiki/"
    res = r.get(base_url + term)

    soup = BeautifulSoup(res.text)
    print(soup.get_text())


if __name__ == "__main__":
    search_term = input("Enter a search term: ")
    main(search_term)