# app.py
import datetime as dt
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import requests
import streamlit as st
import altair as alt
import certifi
from openai import OpenAI
from streamlit_calendar import calendar

# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("체크인 → 달성률/차트 확인 → 날씨/강아지 + AI 코치 리포트 생성")

# =========================
# Session State Init
# =========================
def _init_state():
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = ""
    if "owm_key" not in st.session_state:
        st.session_state.owm_key = ""
    if "records" not in st.session_state:
        st.session_state.records = {}
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_weather" not in st.session_state:
        st.session_state.last_weather = None
    if "last_dog" not in st.session_state:
        st.session_state.last_dog = None
    if "last_extras" not in st.session_state:
        st.session_state.last_extras = {}
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "last_errors" not in st.session_state:
        st.session_state.last_errors = {}
    if "last_selected_sources" not in st.session_state:
        st.session_state.last_selected_sources = []
    if "custom_habits" not in st.session_state:
        st.session_state.custom_habits = []


_init_state()

# =========================
# Constants
# =========================
DEFAULT_HABITS = [
    ("기상 미션", "⏰"),
    ("물 마시기", "💧"),
    ("공부/독서", "📚"),
    ("운동하기", "🏃‍♀️"),
    ("수면", "😴"),
]
DEFAULT_HABIT_KEYS = [h[0] for h in DEFAULT_HABITS]


def _get_habits() -> List[Tuple[str, str]]:
    return DEFAULT_HABITS + st.session_state.custom_habits


def _normalize_habit_records(habit_keys: List[str]) -> None:
    for record in st.session_state.records.values():
        habits = record.get("habits") or {}
        updated = False
        for key in habit_keys:
            if key not in habits:
                habits[key] = False
                updated = True
        if updated:
            record["habits"] = habits


def _achievement_color(achievement_pct: int) -> str:
    if achievement_pct >= 80:
        return "#2ecc71"
    if achievement_pct >= 50:
        return "#f1c40f"
    return "#e74c3c"


def _build_calendar_events(records: Dict[str, Dict[str, Any]], habit_keys: List[str]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    total = len(habit_keys)
    for date_key, rec in records.items():
        habits = rec.get("habits") or {k: False for k in habit_keys}
        achieved_count = sum(1 for v in habits.values() if v)
        achievement_pct = int(round((achieved_count / total) * 100)) if total else 0
        color = _achievement_color(achievement_pct)
        events.append(
            {
                "title": f"{achievement_pct}%",
                "start": date_key,
                "allDay": True,
                "backgroundColor": color,
                "borderColor": color,
                "textColor": "#111111",
            }
        )
    return events


HABITS = _get_habits()
HABIT_KEYS = [h[0] for h in HABITS]

CITIES = ["Seoul", "Busan", "Incheon", "Daegu", "Daejeon", "Gwangju", "Ulsan", "Suwon", "Jeju", "Sejong"]
COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]
API_SOURCES = [
    ("quote", "명언(Quotable)"),
    ("tip", "오늘의 팁(Advice Slip)"),
]
API_SOURCE_LABELS = {key: label for key, label in API_SOURCES}
DEFAULT_API_SOURCE_KEYS = [key for key, _ in API_SOURCES]

STYLE_SYSTEM_PROMPTS = {
    "스파르타 코치": (
        "너는 엄격하지만 정확한 '스파르타 코치'다. 변명은 허용하지 않는다. "
        "짧고 단호하게, 실행 가능한 지시를 준다. 감정적 위로는 최소화한다."
    ),
    "따뜻한 멘토": (
        "너는 공감 능력이 높은 '따뜻한 멘토'다. 비난하지 않고, 작은 성취를 인정하며 "
        "현실적인 다음 행동을 제안한다. 말투는 부드럽고 격려 중심이다."
    ),
    "게임 마스터": (
        "너는 유쾌한 'RPG 게임 마스터'다. 사용자를 플레이어로, 습관을 퀘스트로 표현한다. "
        "레벨업/아이템/버프 같은 게임 요소를 활용해 재미있게 코칭한다."
    ),
}

# =========================
# Sidebar: API Keys (입력 구조 유지 + session_state 저장)
# =========================
with st.sidebar:
    st.header("🔑 API 설정")

    openai_key_in = st.text_input(
        "OpenAI API Key",
        type="password",
        value=st.session_state.openai_key or "",
        help="예: sk-...",
        placeholder="sk-...",
    )
    owm_key_in = st.text_input(
        "OpenWeatherMap API Key",
        type="password",
        value=st.session_state.owm_key or "",
        help="OpenWeatherMap에서 발급받은 키",
        placeholder="OWM key",
    )

    # ✅ 입력값을 session_state에 확실히 저장 (리렌더링/버튼클릭에도 유지)
    if openai_key_in != st.session_state.openai_key:
        st.session_state.openai_key = openai_key_in.strip()
    if owm_key_in != st.session_state.owm_key:
        st.session_state.owm_key = owm_key_in.strip()

    st.divider()
    st.caption("키는 session_state에만 저장됩니다. (배포 시 secrets 권장)")

openai_key = st.session_state.openai_key
owm_key = st.session_state.owm_key

# =========================
# API Helpers
# =========================
def get_weather(city: str, api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    OpenWeatherMap 현재 날씨 (한국어, 섭씨)
    - 실패 시 (None, error_message) 반환
    - timeout=10
    """
    if not api_key:
        return None, "OpenWeatherMap API Key가 비어 있어요."

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": api_key, "units": "metric", "lang": "kr"}

    try:
        r = requests.get(url, params=params, timeout=10)
        # 에러 원인 파악을 위해 status code 기반 메시지 제공
        if r.status_code == 401:
            return None, "OWM 인증 실패(401): API Key가 잘못됐거나 비활성화 상태일 수 있어요."
        if r.status_code == 404:
            return None, "OWM 도시를 찾을 수 없음(404): 도시 이름을 확인해줘요. (예: Seoul)"
        if r.status_code == 429:
            return None, "OWM 호출 제한(429): 잠시 후 다시 시도해줘요."
        r.raise_for_status()

        data = r.json()
        return {
            "city": city,
            "desc": (data.get("weather") or [{}])[0].get("description"),
            "temp": (data.get("main") or {}).get("temp"),
            "feels_like": (data.get("main") or {}).get("feels_like"),
            "humidity": (data.get("main") or {}).get("humidity"),
            "wind": (data.get("wind") or {}).get("speed"),
        }, None
    except requests.Timeout:
        return None, "OWM 요청 시간이 초과됐어요(timeout=10). 네트워크를 확인해줘요."
    except Exception as e:
        return None, f"OWM 오류: {type(e).__name__}"


def _breed_from_dogceo_image_url(image_url: str) -> Optional[str]:
    try:
        breed_raw = image_url.split("/breeds/")[1].split("/")[0]  # e.g., "hound-afghan"
        return breed_raw.replace("-", " ").title()
    except Exception:
        return None


def get_dog_image() -> Tuple[Optional[Tuple[str, Optional[str]]], Optional[str]]:
    """
    Dog CEO 랜덤 이미지 URL + 품종
    - 실패 시 (None, error_message) 반환
    - timeout=10
    """
    url = "https://dog.ceo/api/breeds/image/random"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        image_url = data.get("message")
        if not image_url:
            return None, "Dog CEO 응답에 이미지가 없어요."
        breed = _breed_from_dogceo_image_url(image_url)
        return (image_url, breed), None
    except requests.Timeout:
        return None, "Dog CEO 요청 시간이 초과됐어요(timeout=10)."
    except Exception as e:
        return None, f"Dog CEO 오류: {type(e).__name__}"


def get_quotable_quote() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Quotable 명언
    - 실패 시 (None, error_message)
    - timeout=10
    """
    url = "https://api.quotable.io/random"
    try:
        r = requests.get(
            url,
            timeout=10,
            headers={"Accept": "application/json", "User-Agent": "habit-tracker/1.0"},
            verify=certifi.where(),
        )
        if r.status_code == 429:
            return None, "Quotable 호출 제한(429): 잠시 후 다시 시도해줘요."
        if r.status_code >= 500:
            return None, f"Quotable 서버 오류({r.status_code})가 발생했어요."
        r.raise_for_status()
        data = r.json()
        content = data.get("content")
        if not content:
            return None, "Quotable 응답에 명언이 없어요."
        return {"content": content, "author": data.get("author") or "알 수 없음"}, None
    except requests.Timeout:
        return None, "Quotable 요청 시간이 초과됐어요(timeout=10)."
    except requests.exceptions.SSLError:
        return None, "Quotable SSL 인증 오류가 발생했어요. 네트워크/인증서 상태를 확인해줘요."
    except Exception as e:
        return None, f"Quotable 오류: {type(e).__name__}"


def get_advice_tip() -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """
    Advice Slip 오늘의 팁
    - 실패 시 (None, error_message)
    - timeout=10
    """
    url = "https://api.adviceslip.com/advice"
    try:
        r = requests.get(
            url,
            timeout=10,
            headers={"Accept": "application/json", "User-Agent": "habit-tracker/1.0"},
            verify=certifi.where(),
        )
        if r.status_code == 429:
            return None, "Advice Slip 호출 제한(429): 잠시 후 다시 시도해줘요."
        if r.status_code >= 500:
            return None, f"Advice Slip 서버 오류({r.status_code})가 발생했어요."
        r.raise_for_status()
        data = r.json()
        slip = data.get("slip") or {}
        advice = slip.get("advice")
        if not advice:
            return None, "Advice Slip 응답에 팁이 없어요."
        return {"advice": advice}, None
    except requests.Timeout:
        return None, "Advice Slip 요청 시간이 초과됐어요(timeout=10)."
    except requests.exceptions.SSLError:
        return None, "Advice Slip SSL 인증 오류가 발생했어요. 네트워크/인증서 상태를 확인해줘요."
    except Exception as e:
        return None, f"Advice Slip 오류: {type(e).__name__}"


def generate_report(
    openai_api_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
    extra_sources: Dict[str, Optional[Dict[str, str]]],
) -> Tuple[Optional[str], Optional[str]]:
    """
    OpenAI로 리포트 생성
    - 실패 시 (None, error_message)
    """
    if not openai_api_key:
        return None, "OpenAI API Key가 비어 있어요."

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

    extra_lines: List[str] = []
    if extra_sources:
        for key, label in API_SOURCES:
            if key not in extra_sources:
                continue
            payload = extra_sources.get(key)
            if key == "quote":
                if payload:
                    extra_lines.append(
                        f"- {label}: \"{payload.get('content')}\" — {payload.get('author') or '알 수 없음'}"
                    )
                else:
                    extra_lines.append(f"- {label}: 정보 없음")
            elif key == "tip":
                if payload:
                    extra_lines.append(f"- {label}: {payload.get('advice')}")
                else:
                    extra_lines.append(f"- {label}: 정보 없음")
    if not extra_lines:
        extra_lines.append("- 선택된 외부 API 없음")

    extra_text = "\n".join(extra_lines)

    user_payload = f"""
[오늘 체크인 요약]
- 달성 습관: {", ".join(achieved) if achieved else "없음"}
- 미달성 습관: {", ".join(missed) if missed else "없음"}
- 기분(1~10): {mood}
- 날씨: {weather_text}
- 오늘의 강아지 품종: {dog_text}
- 외부 API 데이터:
{extra_text}

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
""".strip()

    system_prompt = STYLE_SYSTEM_PROMPTS.get(coach_style, STYLE_SYSTEM_PROMPTS["따뜻한 멘토"])

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            temperature=0.7,
        )
        return resp.choices[0].message.content, None
    except Exception as e:
        return None, f"OpenAI 오류: {type(e).__name__}"

# =========================
# Seed Demo Data (6일 + 오늘)
# =========================
def _seed_demo_data_if_needed() -> None:
    if st.session_state.records:
        return

    today = dt.date.today()
    # 최근 6일(오늘 제외) 데모 패턴: (달성개수, 기분)
    pattern = [(3, 7), (4, 6), (2, 5), (5, 8), (1, 4), (4, 7)]
    demo = {}

    for offset, (ach_cnt, mood) in zip(range(6, 0, -1), pattern):
        date_ = today - dt.timedelta(days=offset)
        habits = {k: (idx < ach_cnt) for idx, k in enumerate(HABIT_KEYS)}
        demo[str(date_)] = {
            "date": str(date_),
            "habits": habits,
            "mood": mood,
            "city": "Seoul",
            "coach_style": "따뜻한 멘토",
            "api_sources": DEFAULT_API_SOURCE_KEYS,
        }

    demo[str(today)] = {
        "date": str(today),
        "habits": {k: False for k in HABIT_KEYS},
        "mood": 5,
        "city": "Seoul",
        "coach_style": "따뜻한 멘토",
        "api_sources": DEFAULT_API_SOURCE_KEYS,
    }

    st.session_state.records = demo


_seed_demo_data_if_needed()
_normalize_habit_records(HABIT_KEYS)

today = dt.date.today()
today_key = str(today)
if "selected_date" not in st.session_state:
    st.session_state.selected_date = today_key

selected_key = st.session_state.selected_date
selected_date_obj = dt.date.fromisoformat(selected_key)

# =========================
# Calendar
# =========================
st.subheader("🗓️ 기록 캘린더")
calendar_events = _build_calendar_events(st.session_state.records, HABIT_KEYS)
calendar_options = {
    "initialView": "dayGridMonth",
    "locale": "ko",
    "height": 520,
    "headerToolbar": {"left": "prev,next today", "center": "title", "right": "dayGridMonth"},
    "dayMaxEvents": 2,
}

calendar_state = calendar(events=calendar_events, options=calendar_options)
selected_date_clicked = None
if calendar_state.get("dateClick"):
    selected_date_clicked = calendar_state["dateClick"]["date"]
elif calendar_state.get("eventClick"):
    selected_date_clicked = calendar_state["eventClick"]["event"]["start"]

if selected_date_clicked:
    st.session_state.selected_date = selected_date_clicked
    if selected_date_clicked not in st.session_state.records:
        st.session_state.records[selected_date_clicked] = {
            "date": selected_date_clicked,
            "habits": {k: False for k in HABIT_KEYS},
            "mood": 5,
            "city": "Seoul",
            "coach_style": "따뜻한 멘토",
            "api_sources": DEFAULT_API_SOURCE_KEYS,
        }
    st.rerun()

st.caption("달성률 색상: 🟢 80% 이상 · 🟡 50~79% · 🔴 49% 이하")

# =========================
# Main: Check-in UI
# =========================
st.subheader(f"✅ 체크인 ({selected_key})")

left, right = st.columns([1.1, 0.9], vertical_alignment="top")

# 오늘 레코드 로드
current = st.session_state.records.get(selected_key, {})
current_habits = (current.get("habits") or {k: False for k in HABIT_KEYS}).copy()
current_mood = int(current.get("mood") or 5)
current_city = current.get("city") or "Seoul"
current_style = current.get("coach_style") or "따뜻한 멘토"
current_api_sources = current.get("api_sources") or DEFAULT_API_SOURCE_KEYS

with left:
    st.markdown("**습관 추가**")
    habit_input_cols = st.columns([0.75, 0.25])
    with habit_input_cols[0]:
        new_habit = st.text_input("습관 추가", placeholder="예: 영어 단어 10개", label_visibility="collapsed")
    with habit_input_cols[1]:
        add_habit = st.button("추가", use_container_width=True)

    if add_habit:
        habit_name = new_habit.strip()
        if not habit_name:
            st.warning("추가할 습관 이름을 입력해주세요.")
        elif habit_name in HABIT_KEYS:
            st.info("이미 등록된 습관이에요.")
        else:
            st.session_state.custom_habits.append((habit_name, "✨"))
            updated_keys = DEFAULT_HABIT_KEYS + [h[0] for h in st.session_state.custom_habits]
            _normalize_habit_records(updated_keys)
            st.success(f"'{habit_name}' 습관이 추가됐어요!")
            st.rerun()

    st.markdown("**습관 체크(2열)**")
    c1, c2 = st.columns(2)
    updated_habits: Dict[str, bool] = {}

    for idx, (name, emoji) in enumerate(HABITS):
        target_col = c1 if idx % 2 == 0 else c2
        with target_col:
            updated_habits[name] = st.checkbox(
                f"{emoji} {name}",
                value=bool(current_habits.get(name, False)),
                key=f"habit_{name}",
            )

    st.markdown("---")
    mood = st.slider("🙂 기분 점수 (1~10)", 1, 10, value=current_mood, key="mood_slider")

    c_city, c_style = st.columns(2)
    with c_city:
        city = st.selectbox(
            "🏙️ 도시 선택",
            CITIES,
            index=CITIES.index(current_city) if current_city in CITIES else 0,
            key="city_select",
        )
    with c_style:
        coach_style = st.radio(
            "🎭 코치 스타일",
            COACH_STYLES,
            index=COACH_STYLES.index(current_style) if current_style in COACH_STYLES else 1,
            horizontal=False,
            key="coach_style_radio",
        )

    selected_api_labels = st.multiselect(
        "🔌 리포트에 포함할 외부 API",
        [label for _, label in API_SOURCES],
        default=[API_SOURCE_LABELS[key] for key in current_api_sources if key in API_SOURCE_LABELS],
        help="리포트에 넣을 데이터를 선택하세요. (선택하지 않아도 리포트는 생성됩니다.)",
    )
    selected_api_keys = [key for key, label in API_SOURCES if label in selected_api_labels]

    save_btn = st.button("💾 오늘 기록 저장", use_container_width=True)

    if save_btn:
        st.session_state.records[selected_key] = {
            "date": selected_key,
            "habits": updated_habits,
            "mood": mood,
            "city": city,
            "coach_style": coach_style,
            "api_sources": selected_api_keys,
        }
        st.success("체크인이 저장됐어요!")

with right:
    # 저장값이 있으면 저장값, 없으면 현재 UI값을 사용
    used = st.session_state.records.get(selected_key, {})
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

    st.markdown("**📊 최근 7일 달성률**")
    last7 = [selected_date_obj - dt.timedelta(days=i) for i in range(6, -1, -1)]
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
# Analytics: 전체 기록 기반 지표/차트
# =========================
st.divider()
st.subheader("📊 전체 기록 분석")

records_rows: List[Dict[str, Any]] = []
for date_key, rec in st.session_state.records.items():
    habits = rec.get("habits") or {k: False for k in HABIT_KEYS}
    achieved_count = sum(1 for v in habits.values() if v)
    records_rows.append(
        {
            "date": pd.to_datetime(date_key),
            "achieved_count": achieved_count,
            "achievement_rate": achieved_count / len(HABIT_KEYS),
        }
    )

records_df = pd.DataFrame(records_rows).sort_values("date")
if not records_df.empty:
    records_df["day_index"] = range(1, len(records_df) + 1)
    records_df["cumulative_achieved"] = records_df["achieved_count"].cumsum()
    records_df["cumulative_rate"] = records_df["cumulative_achieved"] / (
        len(HABIT_KEYS) * records_df["day_index"]
    )
    records_df["is_full_success"] = records_df["achieved_count"] == len(HABIT_KEYS)

    date_range = pd.date_range(records_df["date"].min(), records_df["date"].max(), freq="D")
    success_full = (
        records_df.set_index("date")["is_full_success"]
        .reindex(date_range, fill_value=False)
        .astype(bool)
    )

    longest_streak = 0
    running = 0
    for success in success_full:
        if success:
            running += 1
            longest_streak = max(longest_streak, running)
        else:
            running = 0

    current_streak = 0
    for success in reversed(success_full.tolist()):
        if success:
            current_streak += 1
        else:
            break

    records_df["month"] = records_df["date"].dt.to_period("M").dt.to_timestamp()
    monthly_avg = records_df.groupby("month")["achievement_rate"].mean()
    latest_month = monthly_avg.index.max()
    latest_month_avg = monthly_avg.loc[latest_month] if latest_month is not None else 0.0

    summary1, summary2, summary3, summary4, summary5 = st.columns(5)
    summary1.metric("누적 달성 횟수", f"{records_df['cumulative_achieved'].iloc[-1]}회")
    summary2.metric("누적 달성률", f"{records_df['cumulative_rate'].iloc[-1] * 100:.1f}%")
    summary3.metric("가장 긴 연속 달성일", f"{longest_streak}일")
    summary4.metric("현재 스트릭", f"{current_streak}일")
    summary5.metric("이번 달 평균 달성률", f"{latest_month_avg * 100:.1f}%")

    st.markdown("**📈 누적 달성률 추이**")
    cumulative_chart = records_df.set_index("date")["cumulative_rate"] * 100
    st.line_chart(cumulative_chart, height=240)

    st.markdown("**🗓️ 월별 달성 히트맵**")
    heatmap_df = records_df.copy()
    heatmap_df["month_label"] = heatmap_df["date"].dt.strftime("%Y-%m")
    heatmap_df["day"] = heatmap_df["date"].dt.day
    heatmap_df["achievement_pct"] = heatmap_df["achievement_rate"] * 100

    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("day:O", title="일"),
            y=alt.Y("month_label:O", title="월"),
            color=alt.Color(
                "achievement_pct:Q",
                title="달성률(%)",
                scale=alt.Scale(scheme="greens"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="날짜"),
                alt.Tooltip("achievement_pct:Q", title="달성률(%)", format=".1f"),
                alt.Tooltip("achieved_count:Q", title="달성 습관 수"),
            ],
        )
        .properties(height=220)
    )
    st.altair_chart(heatmap, use_container_width=True)
    st.caption("※ 달성일은 하루 모든 습관을 체크한 날로 계산합니다.")
else:
    st.info("전체 기록 분석을 위해 최소 1일의 체크인이 필요해요.")

# =========================
# Results: Weather + Dog + AI Report
# =========================
st.divider()
st.subheader("🧠 AI 코치 컨디션 리포트")

# 키 입력 안내(사이드바 입력 유지)
if not owm_key:
    st.warning("☁️ 날씨를 보려면 사이드바에 OpenWeatherMap API Key를 입력해줘요.")
if not openai_key:
    st.warning("📝 AI 리포트를 생성하려면 사이드바에 OpenAI API Key를 입력해줘요.")

gen_btn = st.button("🚀 컨디션 리포트 생성", type="primary", use_container_width=True)

if gen_btn:
    st.session_state.last_error = None
    st.session_state.last_errors = {}
    st.session_state.last_selected_sources = []

    # ✅ 버튼 누른 순간의 최신 UI값을 레코드에 '자동 반영'
    # (사용자가 저장 버튼을 안 눌렀어도, 생성 버튼으로 바로 리포트 만들 수 있게)
    st.session_state.records[selected_key] = {
        "date": selected_key,
        "habits": updated_habits,
        "mood": mood,
        "city": city,
        "coach_style": coach_style,
        "api_sources": selected_api_keys,
    }

    rec = st.session_state.records[selected_key]
    habits_now = rec["habits"]
    mood_now = int(rec["mood"])
    city_now = rec["city"]
    style_now = rec["coach_style"]
    api_sources_now = rec.get("api_sources") or []
    st.session_state.last_selected_sources = api_sources_now

    with st.spinner("날씨/강아지 데이터를 불러오고 리포트를 생성하는 중..."):
        weather, weather_err = get_weather(city_now, owm_key)
        dog, dog_err = get_dog_image()
        dog_url, dog_breed = (dog if dog else (None, None))
        extras_payload: Dict[str, Optional[Dict[str, str]]] = {}
        extras_errors: Dict[str, str] = {}

        if "quote" in api_sources_now:
            quote, quote_err = get_quotable_quote()
            extras_payload["quote"] = quote
            if quote_err:
                extras_errors["quote"] = quote_err
        if "tip" in api_sources_now:
            tip, tip_err = get_advice_tip()
            extras_payload["tip"] = tip
            if tip_err:
                extras_errors["tip"] = tip_err

        report, report_err = generate_report(
            openai_api_key=openai_key,
            coach_style=style_now,
            habits=habits_now,
            mood=mood_now,
            weather=weather,         # ✅ weather None이어도 리포트는 생성 가능
            dog_breed=dog_breed,
            extra_sources=extras_payload,
        )

    st.session_state.last_weather = weather
    st.session_state.last_dog = {"url": dog_url, "breed": dog_breed}
    st.session_state.last_extras = extras_payload
    st.session_state.last_report = report

    # 에러 메시지 모아서 표시(키 자체는 절대 출력하지 않음)
    errs = {}
    if weather_err:
        errs["weather"] = weather_err
    if dog_err:
        errs["dog"] = dog_err
    errs.update(extras_errors)
    if report_err:
        errs["report"] = report_err
    st.session_state.last_errors = errs
    st.session_state.last_error = "\n".join(errs.values()) if errs else None

# Display last fetched
weather = st.session_state.last_weather
dog_info = st.session_state.last_dog or {}
report = st.session_state.last_report
last_error = st.session_state.last_error
extras_info = st.session_state.last_extras or {}
selected_sources = st.session_state.last_selected_sources or []

last_errors = st.session_state.last_errors or {}
if last_errors:
    error_label_map = {
        "weather": "날씨",
        "dog": "강아지",
        "quote": API_SOURCE_LABELS.get("quote", "명언"),
        "tip": API_SOURCE_LABELS.get("tip", "오늘의 팁"),
        "report": "리포트",
    }
    for key, message in last_errors.items():
        label = error_label_map.get(key, key)
        st.error(f"{label}: {message}")
elif last_error:
    st.error(last_error)

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
        st.warning("날씨 정보를 가져오지 못했어요. (API Key/도시/네트워크/활성화 상태를 확인해주세요.)")

with card2:
    st.markdown("#### 🐶 오늘의 강아지")
    if dog_info.get("url"):
        st.image(dog_info["url"], use_container_width=True)
        st.caption(f"품종: {dog_info.get('breed') or '알 수 없음'}")
    else:
        st.warning("강아지 이미지를 가져오지 못했어요. (Dog CEO/네트워크를 확인해주세요.)")

extra_col1, extra_col2 = st.columns(2, vertical_alignment="top")

with extra_col1:
    st.markdown("#### 💬 오늘의 명언")
    quote = extras_info.get("quote")
    if quote:
        st.info(f"“{quote.get('content')}”\n\n- {quote.get('author') or '알 수 없음'}")
    elif "quote" not in selected_sources:
        st.caption("선택하지 않은 API예요.")
    else:
        st.caption("선택하지 않았거나 데이터를 가져오지 못했어요.")

with extra_col2:
    st.markdown("#### 🧩 오늘의 팁")
    tip = extras_info.get("tip")
    if tip:
        st.info(tip.get("advice") or "정보 없음")
    elif "tip" not in selected_sources:
        st.caption("선택하지 않은 API예요.")
    else:
        st.caption("선택하지 않았거나 데이터를 가져오지 못했어요.")

st.markdown("#### 📝 AI 리포트")
if report:
    st.write(report)
else:
    st.info("아직 리포트가 없어요. 위의 **'컨디션 리포트 생성'** 버튼을 눌러주세요. (OpenAI API Key 필요)")

# Share text
st.markdown("#### 📌 공유용 텍스트")

rec_today = st.session_state.records.get(selected_key, {})
hab_today = rec_today.get("habits") or {k: False for k in HABIT_KEYS}
ach_list = [k for k, v in hab_today.items() if v]
miss_list = [k for k, v in hab_today.items() if not v]
mood_today = int(rec_today.get("mood") or 5)
city_today = rec_today.get("city") or "Seoul"
style_today = rec_today.get("coach_style") or "따뜻한 멘토"

weather_line = "정보 없음"
if weather:
    weather_line = f"{weather.get('desc')}, {weather.get('temp')}°C"

quote_line = "정보 없음"
tip_line = "정보 없음"
if extras_info.get("quote"):
    quote_line = f"{extras_info['quote'].get('content')} — {extras_info['quote'].get('author') or '알 수 없음'}"
if extras_info.get("tip"):
    tip_line = extras_info["tip"].get("advice") or "정보 없음"

share = f"""[AI 습관 트래커 - 오늘 체크인]
- 날짜: {selected_key}
- 도시: {city_today}
- 코치 스타일: {style_today}
- 달성: {", ".join(ach_list) if ach_list else "없음"}
- 미달성: {", ".join(miss_list) if miss_list else "없음"}
- 기분: {mood_today}/10
- 날씨: {weather_line}
- 강아지: {dog_info.get('breed') or "정보 없음"}
- 명언: {quote_line}
- 오늘의 팁: {tip_line}

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
- 현재 앱은 *사이드바 입력 → session_state 저장* 구조입니다.

**2) OpenWeatherMap API Key**
- 날씨 카드에 필요합니다.
- `get_weather(city, api_key)`는 다음 옵션으로 호출합니다:
  - `lang=kr` (한국어)
  - `units=metric` (섭씨)
- 키 발급 직후에는 활성화까지 5~15분(가끔 더) 걸릴 수 있어요.

**3) Dog CEO API**
- 키 없이 사용 가능합니다.
- 실패 시에도 앱은 계속 동작하고, 에러는 화면에 표시됩니다.

**4) Quotable / Advice Slip API**
- 키 없이 사용 가능합니다.
- 체크인 화면의 "외부 API" 선택에서 리포트 포함 여부를 선택할 수 있어요.

**5) 네트워크/요금제/제한**
- 401: 키 오류/비활성
- 404: 도시명 오류
- 429: 호출 제한
- timeout: 네트워크 문제 가능
        """.strip()
    )
