# src/recommend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .geo import haversine_km, clamp


@dataclass(frozen=True)
class UserPrefs:
    radius_km: float = 10.0
    guests: int = 2
    max_price_per_night: Optional[float] = None  # None이면 가격 필터 안 함
    priority: str = "location"  # "wine" | "location" | "budget"
    mix: str = "balanced"       # "drinks" | "food" | "see" | "balanced"


# ---------------------------
# Loading helpers
# ---------------------------
def load_pois(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # normalize text columns
    df["category"] = df["category"].astype(str).str.strip().str.lower()
    df["tags"] = df["tags"].fillna("").astype(str)
    df["price_level"] = df["price_level"].fillna("").astype(str)

    # harden numeric columns (THIS FIXES YOUR ERROR)
    for col in ["lat", "lng", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows with invalid coordinates
    df = df.dropna(subset=["lat", "lng"]).copy()

    # rating fallback
    df["rating"] = df["rating"].fillna(0)

    return df


def load_lodgings(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["tags"] = df["tags"].fillna("").astype(str)

    # harden types (CSV가 약간 깨져도 최대한 복구)
    for col in ["lat", "lng", "price_per_night", "max_guests", "rating"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 필수 숫자 컬럼이 NaN이면 제거 (최소한 안전하게)
    df = df.dropna(subset=["lat", "lng", "price_per_night", "max_guests"]).copy()
    df["rating"] = df["rating"].fillna(0)

    return df



# ---------------------------
# Scoring helpers
# ---------------------------
def _tag_match_score(tags: str, wanted: List[str]) -> float:
    """
    Very simple tag match score: fraction of wanted tags found in tags string.
    tags: "a,b,c" or "a b c" 같은 자유 형식도 허용.
    """
    if not wanted:
        return 0.0
    t = tags.lower().replace(",", " ").split()
    tset = set([x.strip() for x in t if x.strip()])
    hit = sum(1 for w in wanted if w in tset)
    return hit / max(1, len(wanted))


def score_lodging(
    row: pd.Series,
    pois_df: pd.DataFrame,
    prefs: UserPrefs,
) -> float:
    """
    숙소 점수:
    - 평점 (있으면)
    - 예산 적합도 (필터는 따로, 점수는 bonus)
    - 주변 POI 밀도(반경 내 POI 개수) (너 서비스 핵심 가치)
    """
    lat, lng = float(row["lat"]), float(row["lng"])

    # rating score (0~5 -> 0~1)
    rating = float(row["rating"]) if pd.notna(row.get("rating")) else 0.0
    rating_s = clamp(rating / 5.0, 0.0, 1.0)

    # price fit (cheap is better if priority=budget)
    price = float(row["price_per_night"]) if pd.notna(row.get("price_per_night")) else 0.0
    if prefs.max_price_per_night:
        price_fit = clamp(1.0 - (price / prefs.max_price_per_night), 0.0, 1.0)
    else:
        # no budget constraint -> neutral
        price_fit = 0.5

    # POI density within radius
    # (속도: 데이터 적으니 간단히 loop)
    count = 0
    for _, p in pois_df.iterrows():
        d = haversine_km(lat, lng, float(p["lat"]), float(p["lng"]))
        if d <= prefs.radius_km:
            count += 1
    # normalize density (0~20개 기준으로 0~1 클램프)
    density_s = clamp(count / 20.0, 0.0, 1.0)

    # weights by priority
    if prefs.priority == "budget":
        w_rating, w_price, w_density = 0.25, 0.45, 0.30
    elif prefs.priority == "wine":
        w_rating, w_price, w_density = 0.30, 0.20, 0.50
    else:  # location
        w_rating, w_price, w_density = 0.35, 0.20, 0.45

    return w_rating * rating_s + w_price * price_fit + w_density * density_s


def recommend_lodgings(
    lodgings_df: pd.DataFrame,
    pois_df: pd.DataFrame,
    prefs: UserPrefs,
    top_k: int = 2,
) -> pd.DataFrame:
    """
    숙소 후보 top_k 반환.
    - 인원수/예산 필터
    - 점수로 정렬
    """
    df = lodgings_df.copy()

    # filters
    df = df[df["max_guests"] >= prefs.guests]
    if prefs.max_price_per_night is not None:
        df = df[df["price_per_night"] <= prefs.max_price_per_night]

    if df.empty:
        return df

    df["score"] = df.apply(lambda r: score_lodging(r, pois_df, prefs), axis=1)
    df = df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    return df


def recommend_pois_near_lodging(
    pois_df: pd.DataFrame,
    lodging_lat: float,
    lodging_lng: float,
    prefs: UserPrefs,
    per_category: Optional[Dict[str, int]] = None,
    exclude_ids: Optional[set] = None,
) -> pd.DataFrame:

    """
    숙소 기준 POI 추천:
    - 반경 필터
    - 카테고리별 top N (항상 2개 이상 줄 수 있게 설정 가능)
    - 스코어: 거리 + 평점 + 태그(간단)
    """
    if per_category is None:
        per_category = {
            "winery": 3,
            "brewery": 3,
            "distillery": 2,
            "restaurant": 3,
            "sight": 4,
        }

    df = pois_df.copy()
    if exclude_ids:
        df = df[~df["id"].isin(exclude_ids)].copy()


    # distance
    df["dist_km"] = df.apply(
        lambda r: haversine_km(lodging_lat, lodging_lng, float(r["lat"]), float(r["lng"])),
        axis=1,
    )
    df = df[df["dist_km"] <= prefs.radius_km].copy()
    if df.empty:
        return df

    # choose wanted tags by mix
    wanted: List[str] = []
    if prefs.mix == "drinks":
        wanted = ["wine", "tasting", "beer", "whiskey", "tour"]
    elif prefs.mix == "food":
        wanted = ["cozy", "romantic", "fine-dining", "casual"]
    elif prefs.mix == "see":
        wanted = ["beach", "nature", "view", "trail", "landmark"]

    # score components
    # dist: closer is better (0km -> 1, radius -> 0)
    df["dist_s"] = df["dist_km"].apply(lambda d: clamp(1.0 - (d / max(1e-6, prefs.radius_km)), 0.0, 1.0))
    df["rating_s"] = df["rating"].fillna(0).apply(lambda x: clamp(float(x) / 5.0, 0.0, 1.0))
    df["tag_s"] = df["tags"].apply(lambda t: _tag_match_score(str(t), wanted))

    # weights by priority (wine priority -> wineries get a bump later too)
    df["base_score"] = 0.55 * df["dist_s"] + 0.35 * df["rating_s"] + 0.10 * df["tag_s"]

    # extra bump if wine priority and category is winery/distillery/brewery
    if prefs.priority == "wine":
        df["base_score"] = df.apply(
            lambda r: float(r["base_score"]) + (0.08 if r["category"] in {"winery", "brewery", "distillery"} else 0.0),
            axis=1,
        )

    # pick top N per category
    picked = []
    for cat, n in per_category.items():
        sub = df[df["category"] == cat].sort_values("base_score", ascending=False).head(n)
        picked.append(sub)

    out = pd.concat(picked, axis=0).sort_values(["category", "base_score"], ascending=[True, False])
    out = out.reset_index(drop=True)
    return out

def score_pois_pool_near_lodging(
    pois_df: pd.DataFrame,
    lodging_lat: float,
    lodging_lng: float,
    prefs: UserPrefs,
) -> pd.DataFrame:
    """
    숙소 기준: 반경 내 POI 전체 후보 풀 + dist_km/base_score 계산해서 반환.
    (앱에서 exclude/replace 같은 인터랙션을 하려면 '전체 풀'이 필요함)
    """
    df = pois_df.copy()

    # distance
    df["dist_km"] = df.apply(
        lambda r: haversine_km(lodging_lat, lodging_lng, float(r["lat"]), float(r["lng"])),
        axis=1,
    )
    df = df[df["dist_km"] <= prefs.radius_km].copy()
    if df.empty:
        return df

    # wanted tags by mix
    wanted: List[str] = []
    if prefs.mix == "drinks":
        wanted = ["wine", "tasting", "beer", "whiskey", "tour"]
    elif prefs.mix == "food":
        wanted = ["cozy", "romantic", "fine-dining", "casual", "italian"]
    elif prefs.mix == "see":
        wanted = ["beach", "nature", "view", "trail", "landmark"]

    # score components
    df["dist_s"] = df["dist_km"].apply(lambda d: clamp(1.0 - (d / max(1e-6, prefs.radius_km)), 0.0, 1.0))
    df["rating_s"] = df["rating"].fillna(0).apply(lambda x: clamp(float(x) / 5.0, 0.0, 1.0))
    df["tag_s"] = df["tags"].apply(lambda t: _tag_match_score(str(t), wanted))

    df["base_score"] = 0.55 * df["dist_s"] + 0.35 * df["rating_s"] + 0.10 * df["tag_s"]

    # wine-priority bump
    if prefs.priority == "wine":
        df["base_score"] = df.apply(
            lambda r: float(r["base_score"]) + (0.08 if r["category"] in {"winery", "brewery", "distillery"} else 0.0),
            axis=1,
        )

    df = df.sort_values(["category", "base_score"], ascending=[True, False]).reset_index(drop=True)
    return df
