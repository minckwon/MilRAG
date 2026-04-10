"""
국방일보 무기백과 크롤러
- /newsWeb/menuList.do API로 무기백과 하위 카테고리를 동적 조회
- 카테고리별 크롤링 후 /root/Desktop/workspace/kwon/milrag/article/weapon/ 에 저장
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import re
import csv
from pathlib import Path

BASE_URL = "https://kookbang.dema.mil.kr"
MENU_API = f"{BASE_URL}/newsWeb/menuList.do"
WEAPON_PARENT_ID = "ATCE_CTGR_0080000000"

OUTPUT_DIR = Path("/root/Desktop/workspace/kwon/milrag/article/weapon")

# 요청 간 딜레이 (초) - 서버 부하 방지
REQUEST_DELAY = 1.0


CATEGORY_NAME_MAP = {
    "기동":     "maneuver",
    "화력":     "firepower",
    "함정":     "naval_vessel",
    "항공기":   "aircraft",
    "유도·탄약": "guided_munition",
    "방호":     "protection",
    "국방로봇": "defense_robot",
    "지휘통신": "command_communication",
    "감시정찰": "surveillance_reconnaissance",
    "R&D이야기": "RnD",
}


def fetch_categories():
    """무기백과 하위 카테고리 목록을 API에서 동적으로 조회"""
    resp = requests.post(
        MENU_API,
        data={"parent_story_ty_id": WEAPON_PARENT_ID},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    # cateList 또는 최상위 배열에서 무기백과 직속 하위 카테고리만 추출
    items = data if isinstance(data, list) else data.get("cateList", [])
    categories = {}
    for item in items:
        ty_id = item.get("story_ty_id", "")
        ty_nm = item.get("story_ty_nm", "")
        parent_id = item.get("parent_story_ty_id", "")
        # 무기백과(ATCE_CTGR_0080000000)의 직속 자식만 포함
        if parent_id == WEAPON_PARENT_ID and ty_id and ty_nm:
            list_url = f"{BASE_URL}/newsWeb/{ty_id}/list.do"
            english_name = CATEGORY_NAME_MAP.get(ty_nm, ty_nm)
            categories[english_name] = list_url

    return categories


def get_session(list_url):
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": list_url,
    })
    session.get(list_url, timeout=10)
    return session


def get_total_pages(session, list_url):
    """전체 페이지 수 파악"""
    resp = session.get(list_url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    page_links = soup.find_all("a", onclick=re.compile(r"fnLinkPage\((\d+)\)"))
    if page_links:
        page_nums = [
            int(re.search(r"fnLinkPage\((\d+)\)", a["onclick"]).group(1))
            for a in page_links
        ]
        return max(page_nums)

    return 1


def get_article_links(session, list_url, page_index=1):
    """목록 페이지에서 기사 링크 수집 (POST 방식 페이지네이션)"""
    data = {
        "pageIndex": str(page_index),
        "pageUnit": "7",
    }
    resp = session.post(list_url, data=data, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    links = []
    ul = soup.select_one("div.list_wrap ul, ul.list_type")
    if ul is None:
        anchors = soup.find_all("a", href=re.compile(r"/newsWeb/\d{8}/\d+/ATCE_CTGR_\w+/view\.do"))
    else:
        anchors = ul.find_all("a", href=re.compile(r"/newsWeb/\d{8}/\d+/ATCE_CTGR_\w+/view\.do"))

    for a in anchors:
        href = a["href"]
        href = re.sub(r";JSESSIONID[^?#]*", "", href)
        full_url = BASE_URL + href
        links.append(full_url)

    return links


def parse_article(session, url, category):
    """기사 페이지에서 제목과 본문 추출"""
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 제목 추출 - og:title이 가장 신뢰도 높음
    title = ""
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content", "").strip():
        title = og_title["content"].strip()
    else:
        for selector in ["h1.tit", "h1", ".view_title"]:
            tag = soup.select_one(selector)
            if tag and tag.get_text(strip=True):
                title = tag.get_text(strip=True)
                break

    # 본문 추출
    content = ""
    body_tag = soup.select_one("#article_body_view, .article_body_main, .view_content")
    if body_tag:
        for tag in body_tag.find_all(["figure", "script", "style"]):
            tag.decompose()
        content = body_tag.get_text(separator="\n", strip=True)
        # 저작권 표시 이후 불필요한 텍스트 제거
        content = re.sub(r"< 저작권자.*", "", content, flags=re.DOTALL)
        content = content.strip()
        content = re.sub(r"\n{3,}", "\n\n", content)

    # 날짜 추출
    date = ""
    date_tag = soup.select_one(".frst_regist_date, .date, time")
    if date_tag:
        date = date_tag.get_text(strip=True)

    return {
        "url": url,
        "category": category,
        "title": title,
        "date": date,
        "content": content,
    }


def crawl_category(category, list_url):
    """카테고리 하나를 크롤링해서 article 리스트 반환"""
    print(f"\n[{category}] 크롤링 시작: {list_url}")
    session = get_session(list_url)

    total_pages = get_total_pages(session, list_url)
    print(f"  총 페이지 수: {total_pages}")

    all_links = []
    for page in range(1, total_pages + 1):
        print(f"  목록 페이지 {page}/{total_pages} 수집 중...")
        links = get_article_links(session, list_url, page_index=page)
        print(f"    → {len(links)}개 링크 발견")
        all_links.extend(links)
        time.sleep(REQUEST_DELAY)

    all_links = list(dict.fromkeys(all_links))
    print(f"  총 기사 수: {len(all_links)}개")

    articles = []
    for i, url in enumerate(all_links, 1):
        print(f"  기사 {i}/{len(all_links)}: {url}")
        try:
            article = parse_article(session, url, category)
            articles.append(article)
            print(f"    제목: {article['title'][:50]}")
        except Exception as e:
            print(f"    오류 발생: {e}")
            articles.append({"url": url, "category": category, "title": "", "date": "", "content": "", "error": str(e)})
        time.sleep(REQUEST_DELAY)

    return articles


def save_category(category, articles):
    """카테고리별 JSON/CSV 저장"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 파일명에 쓸 수 없는 문자 제거 (예: 유도·탄약 → 유도탄약)
    safe_name = re.sub(r'[\\/*?:"<>|·]', "_", category)

    json_path = OUTPUT_DIR / f"{safe_name}.json"
    csv_path  = OUTPUT_DIR / f"{safe_name}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"  JSON 저장: {json_path}")

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "category", "title", "date", "content"])
        writer.writeheader()
        for a in articles:
            writer.writerow({k: a.get(k, "") for k in ["url", "category", "title", "date", "content"]})
    print(f"  CSV 저장: {csv_path}")


if __name__ == "__main__":
    print("카테고리 목록 조회 중...")
    categories = fetch_categories()

    if not categories:
        print("카테고리를 가져오지 못했습니다. 종료.")
        exit(1)

    print(f"총 {len(categories)}개 카테고리 발견:")
    for name, url in categories.items():
        print(f"  - {name}: {url}")

    total_count = 0
    for category, list_url in categories.items():
        articles = crawl_category(category, list_url)
        save_category(category, articles)
        total_count += len(articles)
        print(f"[{category}] 완료: {len(articles)}개")

    print(f"\n전체 완료! 총 {total_count}개 기사 수집.")
    print(f"저장 위치: {OUTPUT_DIR}")
