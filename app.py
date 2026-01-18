# app.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from src.state import PlanState
from src.recommend import (
    UserPrefs,
    load_lodgings,
    load_pois,
    recommend_lodgings,
    recommend_pois_near_lodging,
)

# --------------------
# Page config + CSS
# --------------------
st.set_page_config(page_title="TC Decision Planner (MVP)", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"]  { font-size: 18px; }

      .tc-title {
        text-align: center;
        font-size: 44px;
        font-weight: 800;
        margin: 10px 0 18px 0;
        letter-spacing: -0.5px;
      }
      .tc-section {
        font-size: 28px;
        font-weight: 800;
        margin: 0 0 10px 0;
      }
      .tc-step {
        font-size: 22px;
        font-weight: 700;
        opacity: 0.9;
        text-align: center;
        margin-top: 2px;
      }
      .tc-question {
        font-size: 32px;
        font-weight: 800;
        margin: 12px 0 10px 0;
      }
      .tc-pickone {
        font-size: 26px;
        font-weight: 800;
        margin: 12px 0 8px 0;
      }

      .stRadio label, .stCheckbox label, .stSelectbox label, .stNumberInput label {
        font-size: 18px !important;
      }
      .stButton button {
        font-size: 16px !important;
        padding: 0.55rem 0.9rem !important;
      }

      /* column gap 줄이기 */
      div[data-testid="stHorizontalBlock"]{
        gap: 1.1rem !important;
      }

      .block-container { max-width: 1400px; padding-top: 1.2rem; }

      .stApp {
        background-color: #025951;
        }

      /* Streamlit 기본 헤더 완전 제거 */
      header[data-testid="stHeader"] {
        display: none;
     }

      /* 상단 여백도 같이 제거 */
      div[data-testid="stToolbar"] {
        display: none;
    }

    section[data-testid="stMain"] {
      padding-top: 0rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="tc-title">Traverse City — Decision-first Trip Planner (MVP)</div>',
    unsafe_allow_html=True,
)

# --------------------
# Session state defaults
# --------------------
if "mode" not in st.session_state:
    st.session_state.mode = "build"  # build | result

if "step" not in st.session_state:
    st.session_state.step = 1

if "plan" not in st.session_state:
    st.session_state.plan = PlanState()

# wizard persisted defaults
if "priority" not in st.session_state:
    st.session_state.priority = "location"
if "radius_km" not in st.session_state:
    st.session_state.radius_km = 10
if "guests" not in st.session_state:
    st.session_state.guests = 2
if "max_price" not in st.session_state:
    st.session_state.max_price = 250
if "mix" not in st.session_state:
    st.session_state.mix = "balanced"
if "selected_stay_idx" not in st.session_state:
    st.session_state.selected_stay_idx = 0


# --------------------
# Helpers
# --------------------
@st.cache_data
def load_data():
    pois = load_pois("data/pois.csv")
    lod = load_lodgings("data/lodgings.csv")
    return pois, lod


def maps_url(lat: float, lng: float) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={lat},{lng}"


def photo_box(title: str, subtitle: str = ""):
    st.markdown(
        f"""
        <div style="border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:14px; margin:8px 0;">
          <div style="font-size:14px; opacity:0.9;"><b>{title}</b></div>
          <div style="font-size:12px; opacity:0.7;">{subtitle}</div>
          <div style="height:120px; margin-top:10px; border-radius:12px;
                      background:linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));">
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_prefs_from_session() -> UserPrefs:
    return UserPrefs(
        radius_km=float(st.session_state.radius_km),
        guests=int(st.session_state.guests),
        max_price_per_night=None if int(st.session_state.max_price) == 0 else float(st.session_state.max_price),
        priority=str(st.session_state.priority),
        mix=str(st.session_state.mix),
    )


def compute_best_lod(pois_df: pd.DataFrame, lodgings_df: pd.DataFrame, prefs: UserPrefs) -> pd.DataFrame:
    best_lod = recommend_lodgings(lodgings_df, pois_df, prefs, top_k=2)
    return best_lod


def render_map_big(
    chosen_lat: float,
    chosen_lng: float,
    rec_pois: pd.DataFrame,
    width: int = 1100,
    key: str = "map_build_main",
    height: int = 700,
):
    st.markdown('<div class="tc-section">Algorithm Map</div>', unsafe_allow_html=True)

    m = folium.Map([chosen_lat, chosen_lng], zoom_start=12)

    folium.Marker(
        [chosen_lat, chosen_lng],
        tooltip="Your stay",
        icon=folium.Icon(color="blue", icon="home", prefix="fa"),
    ).add_to(m)

    color_map = {
        "winery": "purple",
        "brewery": "darkblue",
        "distillery": "red",
        "restaurant": "orange",
        "sight": "green",
    }

    for _, r in rec_pois.iterrows():
        folium.Marker(
            [float(r["lat"]), float(r["lng"])],
            tooltip=str(r["name"]),
            popup=folium.Popup(str(r["name"]), max_width=250),
            icon=folium.Icon(color=color_map.get(str(r["category"]), "gray"), icon="info-sign"),
        ).add_to(m)

    # ✅ width/height/key 다 여기서 컨트롤
    st_folium(m, height=height, width=width, key=key)



def render_map_mini(chosen_lat: float, chosen_lng: float):
    m = folium.Map([chosen_lat, chosen_lng], zoom_start=12)
    folium.Marker([chosen_lat, chosen_lng], tooltip="Stay").add_to(m)
    # KEY 중요: result mini map
    st_folium(m, height=420, key="map_result_mini")


# --------------------
# Load data
# --------------------
pois_df, lodgings_df = load_data()


# =========================================================
# BUILD MODE
# =========================================================
def render_build():
    map_col, panel_col = st.columns([3.5, 1.5], gap="large")

    # -------------------------
    # 1) Panel (right)
    # -------------------------
    with panel_col:
        st.markdown('<div class="tc-section">Decision Tree Builder</div>', unsafe_allow_html=True)

        # Step controls
        c_prev, c_mid, c_next = st.columns([1, 2, 1])
        with c_prev:
            if st.button("⬅ Back", disabled=(st.session_state.step == 1),
                         use_container_width=True, key="btn_back"):
                st.session_state.step -= 1
                st.rerun()
        with c_mid:
            st.markdown(f'<div class="tc-step">Step {st.session_state.step} / 6</div>', unsafe_allow_html=True)
        with c_next:
            if st.button("Next ➡", disabled=(st.session_state.step == 6),
                         use_container_width=True, key="btn_next"):
                st.session_state.step += 1
                st.rerun()

        st.caption("Choose step-by-step. You can always go back and adjust.")

        step = int(st.session_state.step)

        # Step content
        if step == 1:
            st.markdown('<div class="tc-question">1) What matters most?</div>', unsafe_allow_html=True)
            photo_box("Decision: Priority", "Minimize driving vs wine-focused vs budget")
            st.markdown('<div class="tc-pickone">Pick one</div>', unsafe_allow_html=True)

            st.session_state.priority = st.radio(
                label="",
                options=["location", "wine", "budget"],
                index=["location", "wine", "budget"].index(st.session_state.priority),
                format_func=lambda x: {"location": "Less driving", "wine": "Wine-focused", "budget": "Budget"}[x],
                label_visibility="collapsed",
                key="radio_priority",
            )

        elif step == 2:
            st.markdown('<div class="tc-question">2) How far are you willing to go?</div>', unsafe_allow_html=True)
            photo_box("Decision: Radius", "Smaller radius = tighter cluster near your stay")
            st.markdown('<div class="tc-pickone">Pick one</div>', unsafe_allow_html=True)

            st.session_state.radius_km = st.radio(
                label="",
                options=[5, 10, 20],
                index=[5, 10, 20].index(int(st.session_state.radius_km)),
                horizontal=True,
                label_visibility="collapsed",
                key="radio_radius",
            )

        elif step == 3:
            st.markdown('<div class="tc-question">3) Group size</div>', unsafe_allow_html=True)
            photo_box("Decision: Guests", "This affects which stays qualify")

            st.session_state.guests = st.number_input(
                "Guests", 1, 10,
                value=int(st.session_state.guests),
                key="num_guests",
            )

        elif step == 4:
            st.markdown('<div class="tc-question">4) Budget</div>', unsafe_allow_html=True)
            photo_box("Decision: Budget", "Set max price per night or ignore")

            st.session_state.max_price = st.number_input(
                "Max price/night (0 = ignore)", 0, 2000,
                value=int(st.session_state.max_price),
                step=10,
                key="num_budget",
            )

        elif step == 5:
            st.markdown('<div class="tc-question">5) What do you want more of?</div>', unsafe_allow_html=True)
            photo_box("Decision: Mix", "Balanced vs Drinks/Food/Sightseeing emphasis")
            st.markdown('<div class="tc-pickone">Pick one</div>', unsafe_allow_html=True)

            st.session_state.mix = st.radio(
                label="",
                options=["balanced", "drinks", "food", "see"],
                index=["balanced", "drinks", "food", "see"].index(st.session_state.mix),
                format_func=lambda x: {"balanced": "Balanced", "drinks": "Drinks", "food": "Food", "see": "Sightseeing"}[x],
                label_visibility="collapsed",
                key="radio_mix",
            )

        elif step == 6:
            st.markdown('<div class="tc-question">6) Choose your stay</div>', unsafe_allow_html=True)
            photo_box("Decision: Stay", "This becomes your base on the map")

        # prefs computed every rerun
        prefs = get_prefs_from_session()

        best_lod = compute_best_lod(pois_df, lodgings_df, prefs)
        if best_lod.empty:
            st.error("No lodging matches your filters.")
            st.stop()

        stay_labels = [
            f"{r['name']} (${int(r['price_per_night'])}, ⭐{r['rating']})"
            for _, r in best_lod.iterrows()
        ]

        if st.session_state.selected_stay_idx >= len(stay_labels):
            st.session_state.selected_stay_idx = 0

        if step == 6:
            st.session_state.selected_stay_idx = st.radio(
                "Stay options",
                options=list(range(len(stay_labels))),
                index=int(st.session_state.selected_stay_idx),
                format_func=lambda i: stay_labels[i],
                key="radio_stay",
            )
        else:
            st.caption("Stay selection unlocks at Step 6.")

        st.divider()
        if st.button(
            "✅ Generate Plan (Finish)",
            type="primary",
            disabled=(step < 6),
            use_container_width=True,
            key="btn_finish",
        ):
            st.session_state.mode = "result"
            st.rerun()

    # -------------------------
    # 2) Map (left)  ✅ 밖으로 뺌!
    # -------------------------
    prefs = get_prefs_from_session()
    best_lod = compute_best_lod(pois_df, lodgings_df, prefs)
    chosen_stay = best_lod.iloc[int(st.session_state.selected_stay_idx)]
    chosen_lat = float(chosen_stay["lat"])
    chosen_lng = float(chosen_stay["lng"])

    rec_pois = recommend_pois_near_lodging(
        pois_df,
        chosen_lat,
        chosen_lng,
        prefs,
        exclude_ids=st.session_state.plan.excluded_poi_ids,
    )

    with map_col:
        render_map_big(chosen_lat, chosen_lng, rec_pois, width=1200, height=600, key="map_build_main")





# =========================================================
# RESULT MODE
# =========================================================
def render_result():
    st.markdown('<div class="tc-section">Your Plan (Result)</div>', unsafe_allow_html=True)

    prefs = get_prefs_from_session()

    best_lod = compute_best_lod(pois_df, lodgings_df, prefs)
    if best_lod.empty:
        st.error("No lodging found. Go back and rebuild.")
        if st.button("✏️ Edit plan", key="btn_edit_from_error"):
            st.session_state.mode = "build"
            st.rerun()
        st.stop()

    if st.session_state.selected_stay_idx >= len(best_lod):
        st.session_state.selected_stay_idx = 0

    chosen_stay = best_lod.iloc[int(st.session_state.selected_stay_idx)]
    chosen_lat = float(chosen_stay["lat"])
    chosen_lng = float(chosen_stay["lng"])

    rec_pois = recommend_pois_near_lodging(
        pois_df,
        chosen_lat,
        chosen_lng,
        prefs,
        exclude_ids=st.session_state.plan.excluded_poi_ids,
    )

    st.markdown("## Base stay")
    c1, c2 = st.columns([2, 1], gap="small")

    with c1:
        st.markdown(f"### {chosen_stay['name']}")
        st.write(f"Price/night: ${int(chosen_stay['price_per_night'])}")
        st.write(f"Max guests: {int(chosen_stay['max_guests'])}")
        st.write(f"Rating: {chosen_stay['rating']}")
        st.markdown(f"[Open in Google Maps]({maps_url(float(chosen_stay['lat']), float(chosen_stay['lng']))})")

    with c2:
        render_map_mini(chosen_lat, chosen_lng)

    st.divider()
    st.markdown("## Recommended places")

    if rec_pois.empty:
        st.info("No POIs to show (everything excluded or radius too small).")
    else:
        for cat, group in rec_pois.groupby("category"):
            st.markdown(f"### {str(cat).title()}")
            for _, r in group.sort_values("dist_km").iterrows():
                with st.container(border=True):
                    st.markdown(f"**{r['name']}**")
                    st.write(f"Distance: {r['dist_km']:.2f} km • Rating: {r['rating']}")
                    st.write(f"Tags: {r.get('tags', '')}")
                    st.markdown(f"[Open in Google Maps]({maps_url(float(r['lat']), float(r['lng']))})")

    st.divider()
    if st.button("✏️ Edit plan", key="btn_edit_plan"):
        st.session_state.mode = "build"
        st.rerun()


# --------------------
# Main router (NO MIXING)
# --------------------
if st.session_state.mode == "build":
    render_build()
else:
    render_result()
