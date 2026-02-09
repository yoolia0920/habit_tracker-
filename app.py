# app.py
import datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
from openai import OpenAI

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("체크인 → 달성률/차트 확인 → 날씨/강아지 + AI 코치 리포트 생성")

# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔑 API 설정")
    openai_key = st.text_input("OpenAI API Key", type="password", help="예: sk-...")
    owm_key = st.text_input("OpenWeatherMap API Key", type="password", help="OpenWeatherMap에서 발급받은 키")
    st.divider()
    st.caption("키는 session_state에만 보관됩니다. (서버 배포 시 secrets 사용 권장)")

# =========================
# Helpers: API
# =========================
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap에서 현재 날씨를 가져옵니다.
    - 한국어(lang=kr), 섭씨(units=metric)
    - 실패 시 None 반환
    """
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        return {
            "city": city,
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "temp": (data.get("main") or {}).get("temp"),
            "feels_like": (data.get("main") or {}).get("feels_like"),
            "humidity": (data.get("main") or {}).get("humidity"),
            "wind": (data.get("wind") or {}).get("speed"),
        }
    except Exception:
        return None


def _breed_from_dogceo_image_url(image_url: str) -> Optional[str]:
    """
    Dog CEO 이미지 URL에서 품종 추출:
    https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    -> hound-afghan -> Hound Afghan
    """
    try:
        parts = image_url.split("/breeds/")[1].split("/")
        breed_raw = parts[0]  # e.g., "hound-afghan"
        breed = breed_raw.replace("-", " ").title()
        return breed
    except Exception:
        return None


def get_dog_image() -> Optional[Tuple[str, Optional[str]]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 반환
    - 실패 시 None 반환
    """
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        image_url = data.get("message")
        if not image_url:
            return None
        breed = _breed_from_dogceo_image_url(image_url)
        return image_url, breed
    except Exception:
        return None


# =========================
# AI Coach
# =========================
STYLE_SYSTEM_PROMPTS = {
    "스파르타 코치": (
        "너는 엄격하지만 정확한 '스파르타 코치'다. 변명은 허용하지 않는다. "
        "짧고 단호하게, 실행 가능한 지시를 준다. 감정적 위로는 최소화한다."
    ),
    "따뜻한 멘토": (
        "너는 공감 능력이 높은 '따뜻한 멘토'다. 비난하지 않고, 작은 성취를 인정하며 "
        "현실적인 다음 քայլ(행동)을 제안한다. 말투는 부드럽고 격려 중심이다."
    ),
    "게임 마스터": (
        "너는 유쾌한 'RPG 게임 마스터'다. 사용자를 플레이어로, 습관을 퀘스트로 표현한다. "
        "레벨업/아이템/버프 같은 게임 요소를 활용해 재미있게 코칭한다."
    ),
}


def generate_report(
    openai_api_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아 OpenAI에 전달해 리포트를 생성합니다.
    - 실패 시 None 반환
    """
    if not openai_api_key:
        return None

    achieved = [k for k, v in habits.items() if v]
    missed = [k for k, v in habits.items() if not v]

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('desc')} / "
            f"{weather.get('temp')}°C(체감 {weather.get('feels_like')}°C) / "
            f"습도 {weather.get('humidity')}% / 바람 {weather.get('wind')}m/s"
        )

    dog_text = dog_breed or "알 수 없음"

    user_payload = f"""
[오늘 체크인 요약]
- 달성 습관: {", ".join(achieved) if achieved else "없음"}
- 미달성 습관: {", ".join(missed) if missed else "없음"}
- 기분(1~10): {mood}
- 날씨: {weather_text}
- 오늘의 강아지 품종: {dog_text}

[요청 출력 형식]
아래 5개 항목을 반드시 같은 순서로 출력해줘. 각 항목은 한 줄 제목으로 시작하고, 그 아래에 2~5줄로 내용 작성.
1) 컨디션 등급(S~D)
2) 습관 분석
3) 날씨 코멘트
4) 내일 미션
5) 오늘의 한마디

추가 규칙:
- 과장하지 말고, 실행 가능한 조언 위주.
- 한국어로 작성.
"""

    system_prompt = STYLE_SYSTEM_PROMPTS.get(coach_style, STYLE_SYSTEM_PROMPTS["따뜻한 멘토"])

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload.strip()},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception:
        return None


# =========================
# Session State: Records
# =========================
HABITS = [
    ("기상 미션", "⏰"),
    ("물 마시기", "💧"),
    ("공부/독서", "📚"),
    ("운동하기", "🏃‍♀️"),
    ("수면", "😴"),
]
HABIT_KEYS = [h[0] for h in HABITS]

CITIES = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Ulsan", "Suwon", "Jeju", "Sejong"]
COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]

def _seed_demo_data_if_needed() -> None:
    if "records" in st.session_state:
        return

    today = dt.date.today()
    # 데모용 6일 샘플 + 오늘(초기값은 빈 체크인)
    demo = {}
    # 최근 6일 (오늘 제외)
    pattern = [
        (3, 7), (4, 6), (2, 5), (5, 8), (1, 4), (4, 7)
    ]  # (달성개수, 기분)
    for i, (ach_cnt, mood) in enumerate(pattern, start=6):
        d = today - dt.timedelta(days=i)  # 6~11일 전이 아니라 6일치로 맞추기 위해 아래에서 재정렬
    # 정확히 "최근 6일"이 되도록:
    for offset, (ach_cnt, mood) in zip(range(6, 0, -1), pattern):
        date_ = today - dt.timedelta(days=offset)
        # 습관을 앞에서부터 ach_cnt개 True로
        habits = {k: (idx < ach_cnt) for idx, k in enumerate(HABIT_KEYS)}
        demo[str(date_)] = {
            "date": str(date_),
            "habits": habits,
            "mood": mood,
            "city": "Seoul",
            "coach_style": "따뜻한 멘토",
        }

    # 오늘 기본 레코드(체크인 UI 값으로 덮어쓰기 가능)
    demo[str(today)] = {
        "date": str(today),
        "habits": {k: False for k in HABIT_KEYS},
        "mood": 5,
        "city": "Seoul",
        "coach_style": "따뜻한 멘토",
    }

    st.session_state.records = demo
    st.session_state.last_report = None
    st.session_state.last_weather = None
    st.session_state.last_dog = None

_seed_demo_data_if_needed()

today = dt.date.today()
today_key = str(today)

# =========================
# Main: Check-in UI
# =========================
st.subheader("✅ 오늘 체크인")

left, right = st.columns([1.1, 0.9], vertical_alignment="top")

with left:
    st.markdown("**습관 체크(2열)**")

    # 오늘 레코드 초기값 로드
    current = st.session_state.records.get(today_key, {})
    current_habits = (current.get("habits") or {k: False for k in HABIT_KEYS}).copy()
    current_mood = int(current.get("mood") or 5)
    current_city = current.get("city") or "Seoul"
    current_style = current.get("coach_style") or "따뜻한 멘토"

    c1, c2 = st.columns(2)
    updated_habits = {}

    # 5개 체크박스 2열 배치
    for idx, (name, emoji) in enumerate(HABITS):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            updated_habits[name] = st.checkbox(
                f"{emoji} {name}",
                value=bool(current_habits.get(name, False)),
                key=f"habit_{name}",
            )

    st.markdown("---")
    mood = st.slider("🙂 기분 점수 (1~10)", min_value=1, max_value=10, value=current_mood, key="mood_slider")

    c_city, c_style = st.columns(2)
    with c_city:
        city = st.selectbox("🏙️ 도시 선택", CITIES, index=CITIES.index(current_city) if current_city in CITIES else 0)
    with c_style:
        coach_style = st.radio("🎭 코치 스타일", COACH_STYLES, index=COACH_STYLES.index(current_style), horizontal=False)

    save_btn = st.button("💾 오늘 기록 저장", use_container_width=True)

    if save_btn:
        st.session_state.records[today_key] = {
            "date": today_key,
            "habits": updated_habits,
            "mood": mood,
            "city": city,
            "coach_style": coach_style,
        }
        st.success("오늘 체크인이 저장됐어요!")

with right:
    # 달성률 계산
    used = st.session_state.records.get(today_key, {})
    used_habits = used.get("habits") or updated_habits
    used_mood = int(used.get("mood") or mood)

    achieved_count = sum(1 for v in used_habits.values() if v)
    total = len(HABIT_KEYS)
    achievement = int(round((achieved_count / total) * 100))

    st.markdown("**📈 오늘 요약**")
    m1, m2, m3 = st.columns(3)
    m1.metric("달성률", f"{achievement}%")
    m2.metric("달성 습관", f"{achieved_count}/{total}")
    m3.metric("기분", f"{used_mood}/10")

    st.markdown("---")

    # 7일 바 차트 (데모 6일 + 오늘)
    st.markdown("**📊 최근 7일 달성률**")
    # 최근 7일 날짜 키
    last7 = [today - dt.timedelta(days=i) for i in range(6, -1, -1)]
    rows: List[Dict[str, Any]] = []
    for d in last7:
        k = str(d)
        rec = st.session_state.records.get(k)
        if rec:
            hab = rec.get("habits") or {hk: False for hk in HABIT_KEYS}
            cnt = sum(1 for v in hab.values() if v)
            pct = (cnt / total) * 100
            rows.append({"date": k, "achievement_pct": pct})
        else:
            rows.append({"date": k, "achievement_pct": 0.0})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    st.bar_chart(df["achievement_pct"], height=220)


# =========================
# Results: Weather + Dog + AI Report
# =========================
st.divider()
st.subheader("🧠 AI 코치 컨디션 리포트")

gen_btn = st.button("🚀 컨디션 리포트 생성", type="primary", use_container_width=True)

if gen_btn:
    # 최신 저장값 우선 사용
    rec = st.session_state.records.get(today_key, {
        "habits": updated_habits,
        "mood": mood,
        "city": city,
        "coach_style": coach_style,
    })
    habits_now = rec.get("habits") or updated_habits
    mood_now = int(rec.get("mood") or mood)
    city_now = rec.get("city") or city
    style_now = rec.get("coach_style") or coach_style

    with st.spinner("날씨/강아지 데이터를 불러오고 리포트를 생성하는 중..."):
        weather = get_weather(city_now, owm_key)
        dog = get_dog_image()
        dog_url, dog_breed = (dog if dog else (None, None))

        report = generate_report(
            openai_api_key=openai_key,
            coach_style=style_now,
            habits=habits_now,
            mood=mood_now,
            weather=weather,
            dog_breed=dog_breed,
        )

    st.session_state.last_weather = weather
    st.session_state.last_dog = {"url": dog_url, "breed": dog_breed}
    st.session_state.last_report = report

# Display last fetched
weather = st.session_state.get("last_weather")
dog_info = st.session_state.get("last_dog") or {}
report = st.session_state.get("last_report")

card1, card2 = st.columns(2, vertical_alignment="top")

with card1:
    st.markdown("#### ☁️ 오늘의 날씨")
    if weather:
        st.info(
            f"**{weather.get('city')}**\n\n"
            f"- 상태: {weather.get('desc')}\n"
            f"- 기온: {weather.get('temp')}°C (체감 {weather.get('feels_like')}°C)\n"
            f"- 습도: {weather.get('humidity')}%\n"
            f"- 바람: {weather.get('wind')} m/s"
        )
    else:
        st.warning("날씨 정보를 가져오지 못했어요. (OpenWeatherMap API Key/도시/네트워크를 확인해주세요.)")

with card2:
    st.markdown("#### 🐶 오늘의 강아지")
    if dog_info.get("url"):
        st.image(dog_info["url"], use_container_width=True)
        st.caption(f"품종: {dog_info.get('breed') or '알 수 없음'}")
    else:
        st.warning("강아지 이미지를 가져오지 못했어요. (Dog CEO API/네트워크를 확인해주세요.)")

st.markdown("#### 📝 AI 리포트")
if report:
    st.write(report)
else:
    st.info("아직 리포트가 없어요. 위의 **'컨디션 리포트 생성'** 버튼을 눌러주세요. (OpenAI API Key 필요)")

# Share text
st.markdown("#### 📌 공유용 텍스트")
rec_today = st.session_state.records.get(today_key, {})
hab_today = rec_today.get("habits") or {k: False for k in HABIT_KEYS}
ach_list = [k for k, v in hab_today.items() if v]
miss_list = [k for k, v in hab_today.items() if not v]
mood_today = int(rec_today.get("mood") or 5)
city_today = rec_today.get("city") or "Seoul"
style_today = rec_today.get("coach_style") or "따뜻한 멘토"

share = f"""[AI 습관 트래커 - 오늘 체크인]
- 날짜: {today_key}
- 도시: {city_today}
- 코치 스타일: {style_today}
- 달성: {", ".join(ach_list) if ach_list else "없음"}
- 미달성: {", ".join(miss_list) if miss_list else "없음"}
- 기분: {mood_today}/10
- 날씨: {weather.get('desc') + f", {weather.get('temp')}°C" if weather else "정보 없음"}
- 강아지: {dog_info.get('breed') or "정보 없음"}

[AI 리포트]
{report or "(리포트 미생성)"}"""
st.code(share, language="text")

# =========================
# API 안내
# =========================
with st.expander("ℹ️ API 안내 / 설정 팁"):
    st.markdown(
        """
**1) OpenAI API Key**
- 리포트 생성에 필요합니다.
- 로컬 개발 시 환경변수/Streamlit secrets 사용을 권장합니다.

**2) OpenWeatherMap API Key**
- 날씨 카드에 필요합니다.
- `get_weather(city, api_key)`는 다음 옵션으로 호출합니다:
  - `lang=kr` (한국어)
  - `units=metric` (섭씨)

**3) Dog CEO API**
- 키 없이 사용 가능합니다.
- 실패 시 `None`을 반환하도록 처리되어 있습니다.

**4) 네트워크/요금제 관련**
- API가 실패하면(키 누락/권한/호출 제한/네트워크) 날씨/리포트가 비어 보일 수 있어요.
- requests는 `timeout=10`으로 설정되어 있습니다.
        """.strip()
    )
