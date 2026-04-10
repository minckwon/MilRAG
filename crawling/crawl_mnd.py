"""
국방부 국방일보 기사 크롤러 (최신 300건)
URL: https://www.mnd.go.kr/cop/kookbang/kookbangIlboList.do
- 목록: POST pageIndex 파라미터로 페이지네이션 (12건/페이지 → 25페이지)
- 상세: POST /cop/kookbang/kookbangIlboView.do (boardSeq, categoryCode)
- 저장: /root/Desktop/workspace/kwon/milrag/article/mnd/
"""

import requests
from bs4 import BeautifulSoup
import json
import re
import time
import csv
from pathlib import Path

BASE_URL = "https://www.mnd.go.kr"
LIST_URL = f"{BASE_URL}/cop/kookbang/kookbangIlboList.do"
VIEW_URL = f"{BASE_URL}/cop/kookbang/kookbangIlboView.do"

COMMON_PARAMS = {
    "handle":  "dema0003",
    "siteId":  "mnd",
    "id":      "mnd_020101000000",
}

TARGET_COUNT  = 300
ARTICLES_PER_PAGE = 12
REQUEST_DELAY = 1.0

OUTPUT_DIR  = Path("/root/Desktop/workspace/kwon/milrag/article/mnd")
OUTPUT_JSON = OUTPUT_DIR / "mnd_news.json"
OUTPUT_CSV  = OUTPUT_DIR / "mnd_news.csv"


def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": LIST_URL,
    })
    session.get(LIST_URL, params=COMMON_PARAMS, timeout=10)
    return session


def get_article_stubs(session, page_index):
    """목록 페이지에서 boardSeq, 제목, 날짜 추출"""
    data = {**COMMON_PARAMS, "pageIndex": str(page_index)}
    resp = session.post(LIST_URL, data=data, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    stubs = []

    # jf_view('categoryCode','boardSeq') 패턴에서 추출
    for a in soup.find_all("a", href=re.compile(r"jf_view")):
        match = re.search(r"jf_view\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", a.get("href", ""))
        if not match:
            match = re.search(r"jf_view\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", str(a))
        if not match:
            continue

        category_code = match.group(1)
        board_seq     = match.group(2)
        title         = a.get_text(strip=True)

        # 날짜: 같은 부모 블록에서 "YYYY.MM.DD" 패턴 탐색
        date = ""
        parent = a.find_parent(["li", "div", "tr"])
        if parent:
            date_match = re.search(r"\d{4}\.\d{2}\.\d{2}", parent.get_text())
            if date_match:
                date = date_match.group(0)

        stubs.append({
            "board_seq":     board_seq,
            "category_code": category_code,
            "title":         title,
            "date":          date,
        })

    return stubs


def parse_article(session, stub):
    """상세 페이지에서 본문 추출"""
    data = {
        **COMMON_PARAMS,
        "boardSeq":    stub["board_seq"],
        "categoryCode": stub["category_code"],
    }
    resp = session.post(VIEW_URL, data=data, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # 제목: detail 페이지의 td.title_box에서 추출
    title = ""
    title_tag = soup.select_one("td.title_box")
    if title_tag and title_tag.get_text(strip=True):
        title = title_tag.get_text(strip=True)

    # 날짜: td.write_date 우선, 없으면 텍스트에서 탐색
    date = stub["date"]
    date_tag = soup.select_one("td.write_date")
    if date_tag:
        date_match = re.search(r"\d{4}[.\-]\d{2}[.\-]\d{2}", date_tag.get_text())
        if date_match:
            date = date_match.group(0)
    elif not date:
        date_match = re.search(r"\d{4}[.\-]\d{2}[.\-]\d{2}", soup.get_text())
        if date_match:
            date = date_match.group(0)

    # 본문
    content = ""
    for selector in [".board_view", ".view_content", ".article_body", "#article_body_view",
                     ".cont", ".bbs_content", "div.content", "td.content"]:
        body_tag = soup.select_one(selector)
        if body_tag:
            for tag in body_tag.find_all(["script", "style"]):
                tag.decompose()
            content = body_tag.get_text(separator="\n", strip=True)
            break

    # 본문 정제: 상단 메타데이터(작성자/작성일/조회수 블록) 및 하단 푸터 제거
    # 패턴: "제목\n작성자 :\n관리자\n작성일 :\nDATE\n조회수 :\nNUMBER\n실제본문..."
    content = re.sub(
        r"^.*?조회수\s*:\s*\n[\d,]+\s*\n?",
        "",
        content,
        count=1,
        flags=re.DOTALL,
    )
    content = re.sub(r"\s*목록으로\s*$", "", content, flags=re.DOTALL)
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = content.strip()

    url = f"{VIEW_URL}?boardSeq={stub['board_seq']}&categoryCode={stub['category_code']}&{('&'.join(f'{k}={v}' for k, v in COMMON_PARAMS.items()))}"

    return {
        "url":      url,
        "title":    title,
        "date":     date,
        "content":  content,
    }


def save_results(articles):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    print(f"JSON 저장: {OUTPUT_JSON}")

    with open(OUTPUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "title", "date", "content"])
        writer.writeheader()
        for a in articles:
            writer.writerow({k: a.get(k, "") for k in ["url", "title", "date", "content"]})
    print(f"CSV 저장: {OUTPUT_CSV}")


def main():
    print(f"국방부 국방일보 크롤링 시작 (목표: {TARGET_COUNT}건)")
    session = get_session()

    total_pages = -(-TARGET_COUNT // ARTICLES_PER_PAGE)  # ceiling division → 25
    print(f"수집 페이지 수: {total_pages}")

    # 1단계: 목록에서 stub 수집
    all_stubs = []
    for page in range(1, total_pages + 1):
        print(f"  목록 페이지 {page}/{total_pages} 수집 중...")
        stubs = get_article_stubs(session, page)
        print(f"    → {len(stubs)}건 발견")
        all_stubs.extend(stubs)
        if len(all_stubs) >= TARGET_COUNT:
            break
        time.sleep(REQUEST_DELAY)

    all_stubs = all_stubs[:TARGET_COUNT]
    print(f"\n총 {len(all_stubs)}건 상세 크롤링 시작\n")

    # 2단계: 상세 페이지 크롤링
    articles = []
    for i, stub in enumerate(all_stubs, 1):
        print(f"  [{i}/{len(all_stubs)}] boardSeq={stub['board_seq']}  {stub['title'][:50]}")
        try:
            article = parse_article(session, stub)
            articles.append(article)
        except Exception as e:
            print(f"    오류: {e}")
            articles.append({
                "url":     f"{VIEW_URL}?boardSeq={stub['board_seq']}",
                "title":   stub["title"],
                "date":    stub["date"],
                "content": "",
                "error":   str(e),
            })
        time.sleep(REQUEST_DELAY)

    save_results(articles)
    print(f"\n완료! 총 {len(articles)}건 수집.")
    print(f"저장 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
