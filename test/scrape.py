import requests
from bs4 import BeautifulSoup

url = "https://www.otoclubturkiye.com/forum/topic/315913-ka-cam-titremesi-ve-ses-%C3%A7%C4%B1karma-sorunu/"
resp = requests.get(url)
soup = BeautifulSoup(resp.text, "html.parser")

comments = soup.find_all("article", class_="cPost ipsBox ipsResponsive_pull ipsComment")

for i, comment in enumerate(comments, start=1):
    # Yazar bilgisi
    author_section = comment.find("h3", class_="ipsType_sectionHead cAuthorPane_author")
    author = author_section.get_text(strip=True) if author_section else "Bilinmiyor"

    # Yorum metni
    content_section = comment.find("div", attrs={"data-role": "commentContent"})
    content = content_section.get_text(strip=True) if content_section else ""

    print(f"{i}. Yazar: {author}")
    print("Yorum:", content)
    print("-" * 50)
