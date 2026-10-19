import json
from pathlib import Path
from urllib.parse import quote, quote_plus

import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="CineMatch | Discover movies", page_icon="🎬", layout="wide")

# Cinemeta is the only online movie search provider. No API key is required.
CINEMETA_SEARCH_URL = "https://v3-cinemeta.strem.io/catalog/movie/top/search={query}.json"
CINEMETA_META_URL = "https://v3-cinemeta.strem.io/meta/movie/{imdb_id}.json"
POSTER_FALLBACK = "https://placehold.co/500x750/171824/e8e2ff?text=Poster+unavailable"

SAMPLE_MOVIES = [
    {"id":"tt1375666", "title":"Inception", "year":2010, "genres":["Sci-Fi","Action","Thriller"], "description":"A thief enters dreams to steal secrets.", "rating":8.8, "poster":POSTER_FALLBACK},
    {"id":"tt0816692", "title":"Interstellar", "year":2014, "genres":["Sci-Fi","Drama","Adventure"], "description":"Astronauts travel through space to save humanity.", "rating":8.7, "poster":POSTER_FALLBACK},
    {"id":"tt0468569", "title":"The Dark Knight", "year":2008, "genres":["Action","Crime","Drama"], "description":"Batman confronts a dangerous criminal in Gotham.", "rating":9.0, "poster":POSTER_FALLBACK},
    {"id":"tt0332280", "title":"The Notebook", "year":2004, "genres":["Romance","Drama"], "description":"A couple experiences a lasting romantic relationship.", "rating":7.8, "poster":POSTER_FALLBACK},
    {"id":"tt1119646", "title":"The Hangover", "year":2009, "genres":["Comedy"], "description":"Friends search for answers after a wild night.", "rating":7.7, "poster":POSTER_FALLBACK},
    {"id":"tt5052448", "title":"Get Out", "year":2017, "genres":["Horror","Thriller","Mystery"], "description":"A visitor discovers disturbing secrets at a family home.", "rating":7.8, "poster":POSTER_FALLBACK},
    {"id":"tt0133093", "title":"The Matrix", "year":1999, "genres":["Sci-Fi","Action"], "description":"A hacker discovers reality is a computer simulation.", "rating":8.7, "poster":POSTER_FALLBACK},
    {"id":"tt3783958", "title":"La La Land", "year":2016, "genres":["Romance","Drama","Music"], "description":"A musician and actor pursue their dreams.", "rating":8.0, "poster":POSTER_FALLBACK},
    {"id":"tt0829482", "title":"Superbad", "year":2007, "genres":["Comedy"], "description":"Two friends navigate an eventful high school night.", "rating":7.6, "poster":POSTER_FALLBACK},
    {"id":"tt6644200", "title":"A Quiet Place", "year":2018, "genres":["Horror","Thriller","Sci-Fi"], "description":"A family survives in a world where sound is dangerous.", "rating":7.5, "poster":POSTER_FALLBACK},
    {"id":"tt2911666", "title":"John Wick", "year":2014, "genres":["Action","Thriller","Crime"], "description":"A retired assassin returns to the criminal underworld.", "rating":7.4, "poster":POSTER_FALLBACK},
    {"id":"tt2543164", "title":"Arrival", "year":2016, "genres":["Sci-Fi","Drama","Mystery"], "description":"A linguist tries to communicate with aliens.", "rating":7.9, "poster":POSTER_FALLBACK},
    {"id":"tt0266543", "title":"Finding Nemo", "year":2003, "genres":["Animation","Adventure","Comedy"], "description":"A clownfish crosses the ocean to find his son.", "rating":8.2, "poster":POSTER_FALLBACK},
    {"id":"tt0120338", "title":"Titanic", "year":1997, "genres":["Romance","Drama"], "description":"A romance unfolds aboard a famous ocean liner.", "rating":7.9, "poster":POSTER_FALLBACK},
    {"id":"tt1457767", "title":"The Conjuring", "year":2013, "genres":["Horror","Mystery","Thriller"], "description":"Paranormal investigators help a troubled family.", "rating":7.5, "poster":POSTER_FALLBACK},
    {"id":"tt0114709", "title":"Toy Story", "year":1995, "genres":["Animation","Comedy","Adventure"], "description":"Toys come to life when humans are not around.", "rating":8.3, "poster":POSTER_FALLBACK},
]

DATA_DIR = Path("data")
USER_FILE = DATA_DIR / "user_data.json"
LEGACY_FILE = DATA_DIR / "ratings.json"


def as_text(value):
    return value.strip() if isinstance(value, str) else ""


def extract_items(payload):
    """Accept IMDbOT and common nested JSON response formats."""
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    # IMDbOT has used more than one response shape over time.
    for key in ("d", "results", "items", "data", "titles", "movies", "result", "metas", "meta"):
        value = payload.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = extract_items(value)
            if nested:
                return nested
    return []


def _first_text(*values):
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def normalise_movie(item):
    if not isinstance(item, dict):
        return None

    title_value = item.get("titleText")
    if isinstance(title_value, dict):
        title_value = title_value.get("text")
    title = _first_text(
        item.get("title"), item.get("name"), item.get("l"),
        item.get("primaryTitle"), title_value, item.get("originalTitle"), item.get("#TITLE")
    )
    if not title:
        return None

    imdb_id = _first_text(item.get("id"), item.get("imdb_id"), item.get("imdbID"), item.get("titleId"), item.get("#IMDB_ID"))
    if not imdb_id:
        # Stable title-derived ID only when the API omitted an ID.
        imdb_id = "api-" + "-".join(title.casefold().split())

    year_value = item.get("year") or item.get("y") or item.get("releaseYear") or item.get("release_date") or item.get("startYear") or item.get("releaseInfo") or item.get("#YEAR")
    if isinstance(year_value, dict):
        year_value = year_value.get("year")
    try:
        year = int(str(year_value)[:4]) if year_value else 0
    except (ValueError, TypeError):
        year = 0

    image = (item.get("poster") or item.get("poster_url") or item.get("imageUrl")
             or item.get("image") or item.get("primaryImage") or item.get("i") or item.get("#IMG_POSTER"))
    if isinstance(image, dict):
        image = image.get("imageUrl") or image.get("url") or image.get("src")
    if isinstance(image, str) and image.startswith("//"):
        image = "https:" + image
    if not isinstance(image, str) or not image.startswith("http"):
        image = POSTER_FALLBACK

    genres = item.get("genres") or item.get("genre") or item.get("genresText") or item.get("#GENRES") or []
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.replace("/", ",").split(",") if g.strip()]
    if not isinstance(genres, list):
        genres = []
    genres = [str(g) for g in genres if str(g).strip()][:4] or ["Other"]

    description = _first_text(item.get("description"), item.get("overview"), item.get("plot"), item.get("s"), item.get("#DESCRIPTION"))
    if not description:
        description = "Movie details were not provided by the search API."

    rating_value = item.get("rating") or item.get("imdbRating") or item.get("aggregateRating") or item.get("averageRating") or item.get("imdbRating") or item.get("#RATING") or 0
    if isinstance(rating_value, dict):
        rating_value = rating_value.get("ratingValue", 0)
    try:
        rating = float(rating_value)
        if rating > 10 or rating < 0:
            rating = 0.0
    except (TypeError, ValueError):
        rating = 0.0

    return {
        "id": imdb_id, "title": title, "year": year, "genres": genres,
        "description": description, "rating": rating, "poster": image,
        "source": "Cinemeta"
    }


@st.cache_data(ttl=3600, show_spinner=False)
def cinemeta_poster(imdb_id):
    """Try Cinemeta's movie metadata endpoint when search has no poster."""
    if not isinstance(imdb_id, str) or not imdb_id.startswith("tt"):
        return ""
    try:
        response = requests.get(
            CINEMETA_META_URL.format(imdb_id=quote(imdb_id, safe="")),
            timeout=5,
            headers={"User-Agent": "Mozilla/5.0 CineMatch/1.0", "Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        poster = meta.get("poster") or meta.get("background") or ""
        if isinstance(poster, dict):
            poster = poster.get("url") or poster.get("src") or ""
        return poster if isinstance(poster, str) and poster.startswith("http") else ""
    except (requests.RequestException, ValueError, TypeError, AttributeError):
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def api_search(query):
    """Search Cinemeta's keyless movie catalogue worldwide."""
    query = query.strip()
    if not query:
        return []

    headers = {"User-Agent": "Mozilla/5.0 CineMatch/1.0", "Accept": "application/json"}
    url = CINEMETA_SEARCH_URL.format(query=quote(query, safe=""))
    try:
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError, TypeError):
        return []

    collected = []
    for item in extract_items(payload):
        movie = normalise_movie(item)
        if movie:
            movie["source"] = "Cinemeta"
            # Search results sometimes omit poster artwork; ask Cinemeta's own
            # metadata endpoint for the same IMDb title before using a placeholder.
            if movie.get("poster") == POSTER_FALLBACK:
                recovered_poster = cinemeta_poster(str(movie.get("id", "")))
                if recovered_poster:
                    movie["poster"] = recovered_poster
            collected.append(movie)

    # Remove duplicate entries while preserving the API's result order.
    result, seen = [], set()
    for movie in collected:
        key = (str(movie.get("id", "")), movie["title"].casefold(), movie.get("year", 0))
        if key not in seen:
            seen.add(key)
            result.append(movie)
    return result[:40]


def load_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {}
    if USER_FILE.exists():
        try:
            data = json.loads(USER_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            data = {}
    elif LEGACY_FILE.exists():
        try:
            data["ratings"] = json.loads(LEGACY_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            data = {}

    def clean_ids(values):
        if not isinstance(values, list):
            return []
        cleaned = []
        for value in values:
            if isinstance(value, (str, int)) and str(value).strip() and str(value) not in cleaned:
                cleaned.append(str(value))
        return cleaned

    ratings = {}
    raw = data.get("ratings", {})
    if isinstance(raw, dict):
        for mid, score in raw.items():
            if isinstance(score, int) and not isinstance(score, bool) and 1 <= score <= 5:
                ratings[str(mid)] = score
    saved_movies = data.get("movies", [])
    if not isinstance(saved_movies, list):
        saved_movies = []
    saved_movies = [normalise_movie(item) for item in saved_movies if isinstance(item, dict)]
    saved_movies = [movie for movie in saved_movies if movie]
    return {"ratings": ratings, "favorites": clean_ids(data.get("favorites", [])), "watchlist": clean_ids(data.get("watchlist", [])), "movies": saved_movies}


def save_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"ratings": st.session_state.ratings, "favorites": st.session_state.favorites, "watchlist": st.session_state.watchlist, "movies": st.session_state.get("online_movies", [])}
    temp = USER_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(USER_FILE)


def toggle_list(name, movie_id):
    movie_id = str(movie_id)
    values = st.session_state[name]
    if movie_id in values:
        values.remove(movie_id)
    else:
        values.append(movie_id)
    save_data()


def make_catalog(extra=None):
    merged = {str(m["id"]): dict(m) for m in SAMPLE_MOVIES}
    for movie in (extra or []):
        if movie and movie.get("id"):
            mid = str(movie["id"])
            # Keep richer sample metadata when the same title appears in both sources.
            if mid not in merged:
                merged[mid] = dict(movie)
    return list(merged.values())


def recommend(movies, favorite_genres, ratings, limit=8):
    if not movies:
        return []
    docs = [" ".join(m.get("genres", [])) + " " + m.get("description", "") + " " + m.get("title", "") for m in movies]
    try:
        matrix = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(docs)
        similarity = cosine_similarity(matrix)
    except ValueError:
        similarity = [[0.0 for _ in movies] for _ in movies]
    indices = {str(m["id"]): i for i, m in enumerate(movies)}
    rated_indices = [(indices[mid], score) for mid, score in ratings.items() if mid in indices]
    liked_indices = [indices[mid] for mid, score in ratings.items() if mid in indices and score >= 4]
    selected_genres = set(favorite_genres)
    results = []
    for i, movie in enumerate(movies):
        mid = str(movie["id"])
        if mid in ratings:
            continue
        genres = set(movie.get("genres", []))
        genre_score = len(genres & selected_genres) / len(selected_genres) if selected_genres else 0.0
        if rated_indices:
            weighted = total = 0.0
            for j, score in rated_indices:
                sim = float(similarity[i][j])
                weighted += sim * ((score - 3) / 2)
                total += sim
            rating_score = max(0.0, min(1.0, (weighted / total + 1) / 2)) if total else 0.5
        else:
            rating_score = 0.5
        content_score = max((float(similarity[i][j]) for j in liked_indices), default=0.0)
        score = 0.45 * genre_score + 0.35 * rating_score + 0.20 * content_score
        results.append({**movie, "score": round(score * 100), "genre_score": genre_score})
    results.sort(key=lambda m: (m["genre_score"] if selected_genres else 0, m["score"], m.get("rating", 0)), reverse=True)
    return results[:limit]


st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at top left,rgba(95,62,145,.22),transparent 36%),#10111a;color:#f5f5fa}
[data-testid="stSidebar"]{background:#171824;border-right:1px solid #2c2d3b}
.hero{padding:1.6rem 1.8rem;border:1px solid #373248;border-radius:22px;background:linear-gradient(120deg,rgba(91,60,140,.65),rgba(25,28,44,.95));margin-bottom:1.3rem}
.hero h1{font-size:clamp(2rem,4vw,3.1rem);margin:0}.hero p{color:#d5d3e2;margin:.5rem 0 0}.eyebrow{color:#d2b6ff;text-transform:uppercase;letter-spacing:.15em;font-size:.76rem;font-weight:700}
.movie-meta{color:#b8b9ca;font-size:.9rem}.pill{display:inline-block;margin:2px 4px 2px 0;padding:3px 8px;border-radius:999px;background:#302942;color:#e0d1ff;font-size:.75rem}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#343544;background:rgba(25,27,40,.78);border-radius:16px}
div.stButton>button{border-radius:10px;font-weight:600;min-height:2.4rem}
</style>""", unsafe_allow_html=True)

if "ratings" not in st.session_state:
    loaded = load_data()
    st.session_state.ratings = loaded["ratings"]
    st.session_state.favorites = loaded["favorites"]
    st.session_state.watchlist = loaded["watchlist"]
    st.session_state.online_movies = loaded.get("movies", [])
if "favorite_genres" not in st.session_state:
    st.session_state.favorite_genres = []
if "online_movies" not in st.session_state:
    st.session_state.online_movies = []


def movie_card(movie, prefix="movie", show_score=False):
    mid = str(movie["id"])
    fav = mid in st.session_state.favorites
    saved = mid in st.session_state.watchlist
    with st.container(border=True):
        poster = movie.get("poster") or POSTER_FALLBACK
        # Recover artwork for built-in sample movies using their real IMDb IDs.
        if poster == POSTER_FALLBACK and str(movie.get("id", "")).startswith("tt"):
            recovered = cinemeta_poster(str(movie["id"]))
            if recovered:
                poster = recovered
        st.image(poster, use_container_width=True)
        st.markdown(f"### {movie['title']}")
        year = movie.get("year") or "Year unknown"
        rating = movie.get("rating", 0)
        rating_text = f"⭐ {rating:.1f}/10" if isinstance(rating, (int, float)) and rating > 0 else "⭐ Rating unavailable"
        st.markdown(f"<div class='movie-meta'>{year} · {rating_text}</div>", unsafe_allow_html=True)
        st.markdown(" ".join(f"<span class='pill'>{g}</span>" for g in movie.get("genres", ["Other"])), unsafe_allow_html=True)
        st.write(movie.get("description") or "No description available.")
        if movie.get("source") == "Cinemeta":
            st.caption(f"Online result · {movie.get('source')}")
        if show_score:
            st.progress(max(0, min(1, movie.get("score", 0) / 100)), text=f"Recommendation match: {movie.get('score', 0)}%")
        if mid in st.session_state.ratings:
            st.caption(f"Your rating: {st.session_state.ratings[mid]}/5 ★")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("♥ Favorited" if fav else "♡ Favorite", key=f"{prefix}_fav_{mid}", use_container_width=True):
                toggle_list("favorites", mid)
                st.rerun()
        with c2:
            if st.button("✓ Saved" if saved else "+ Watchlist", key=f"{prefix}_watch_{mid}", use_container_width=True):
                toggle_list("watchlist", mid)
                st.rerun()


def movie_grid(movies, prefix="grid", show_score=False, ncols=4):
    if not movies:
        st.info("No movies found. Try another search or change the filters.")
        return
    cols = st.columns(ncols)
    for i, movie in enumerate(movies):
        with cols[i % ncols]:
            movie_card(movie, prefix, show_score)


with st.sidebar:
    st.markdown("## 🎬 CineMatch")
    st.caption("Your personal movie finder")
    page = st.radio("Navigation", ["Discover", "For You", "Favorites", "Watchlist", "Rate Movies", "My Ratings"])
    st.divider()
    st.caption("Online movie search: Cinemeta · No API key required")
    st.metric("Movies rated", len(st.session_state.ratings))
    st.metric("Favorites", len(st.session_state.favorites))
    st.metric("Watchlist", len(st.session_state.watchlist))
    if st.button("Reset all my data"):
        st.session_state.ratings, st.session_state.favorites, st.session_state.watchlist = {}, [], []
        save_data()
        st.rerun()

catalog = make_catalog(st.session_state.online_movies)
movie_by_id = {str(m["id"]): m for m in catalog}
all_genres = sorted({g for m in catalog for g in m.get("genres", []) if g != "Other"})

if page == "Discover":
    st.markdown("<div class='hero'><div class='eyebrow'>Your personal cinema guide</div><h1>Find your next favorite movie.</h1><p>Search movies from any country, language, release year, or genre. Save favorites and build a watchlist.</p></div>", unsafe_allow_html=True)
    search = st.text_input("🔎 Search movies worldwide", placeholder="Try Dune, Pushpa, Parasite, Your Name, or an older classic...")
    online_results = []
    if search.strip():
        with st.spinner("Searching the online movie catalog..."):
            online_results = api_search(search.strip())
        if online_results:
            # Keep online movies locally so favorites/watchlist/rating IDs remain resolvable.
            known = {str(m["id"]): m for m in st.session_state.online_movies}
            for m in online_results:
                known[str(m["id"])] = m
            st.session_state.online_movies = list(known.values())[-300:]
            save_data()
            st.success(f"Found {len(online_results)} online result(s).")
        else:
            st.warning("No live results were returned. Showing matching saved/sample movies instead. Cinemeta may be temporarily unavailable; please try again later.")
            st.markdown(f"[Try this title on IMDb](https://www.imdb.com/find/?q={quote_plus(search.strip())})")

    catalog = make_catalog(st.session_state.online_movies)
    selected = st.multiselect("Filter by genre", sorted({g for m in catalog for g in m.get("genres", [])}), default=st.session_state.favorite_genres)
    st.session_state.favorite_genres = selected
    if search.strip():
        query = search.casefold()
        combined = list(catalog)
        known_ids = {str(m["id"]) for m in combined}
        for movie in online_results:
            if str(movie["id"]) not in known_ids:
                combined.append(movie)
                known_ids.add(str(movie["id"]))
        filtered = [
            m for m in combined
            if query in m["title"].casefold()
            or query in m.get("description", "").casefold()
            or any(query in g.casefold() for g in m.get("genres", []))
        ]
    else:
        filtered = SAMPLE_MOVIES + st.session_state.online_movies
    # During a title search, do not let the genre filter hide movies from
    # another country/language or movies whose genre metadata is unavailable.
    # Genre filters still work when browsing without a search query.
    if selected and not search.strip():
        filtered = [m for m in filtered if set(m.get("genres", [])) & set(selected)]
    filtered.sort(key=lambda m: m.get("rating", 0), reverse=True)
    st.subheader(f"🍿 Browse Movies · {len(filtered)} results")
    if not search.strip():
        st.caption("Search uses Cinemeta's movie catalogue and does not add a country filter. Results depend on Cinemeta's available catalogue. Genre filters apply when browsing without a search query.")
    movie_grid(filtered, "discover", ncols=4)

elif page == "For You":
    st.markdown("<div class='hero'><div class='eyebrow'>Personalized picks</div><h1>Made for your taste.</h1><p>Recommendations adapt to genres you choose and movies you rate.</p></div>", unsafe_allow_html=True)
    selected = st.multiselect("Choose your favorite genres", all_genres, default=st.session_state.favorite_genres)
    st.session_state.favorite_genres = selected
    catalog = make_catalog(st.session_state.online_movies)
    if not selected and not st.session_state.ratings:
        st.info("Choose some genres or rate a few movies to personalize your recommendations. Search online titles in Discover first to add them to this session.")
    movie_grid(recommend(catalog, selected, st.session_state.ratings), "recommend", show_score=True, ncols=4)

elif page == "Favorites":
    st.markdown("<div class='hero'><div class='eyebrow'>Your collection</div><h1>♥ Favorites</h1><p>All the movies you have marked as favorites.</p></div>", unsafe_allow_html=True)
    catalog = make_catalog(st.session_state.online_movies)
    movie_grid([m for mid in st.session_state.favorites for m in catalog if str(m["id"]) == mid], "favorites", ncols=4)

elif page == "Watchlist":
    st.markdown("<div class='hero'><div class='eyebrow'>Save for later</div><h1>📚 Your watchlist</h1><p>Your saved movies stay here after refreshing the page when their details are in the local catalog.</p></div>", unsafe_allow_html=True)
    catalog = make_catalog(st.session_state.online_movies)
    movie_grid([m for mid in st.session_state.watchlist for m in catalog if str(m["id"]) == mid], "watchlist", ncols=4)

elif page == "Rate Movies":
    st.markdown("<div class='hero'><div class='eyebrow'>Teach CineMatch your taste</div><h1>⭐ Rate a movie</h1><p>Your ratings help improve recommendations.</p></div>", unsafe_allow_html=True)
    catalog = make_catalog(st.session_state.online_movies)
    title_to_movie = {f"{m['title']} ({m.get('year') or 'year unknown'}) · {m['id']}": m for m in catalog}
    choice = st.selectbox("Select a movie", list(title_to_movie.keys()))
    movie = title_to_movie[choice]
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(movie.get("poster") or POSTER_FALLBACK, use_container_width=True)
    with c2:
        st.subheader(movie["title"])
        st.write(movie.get("description", "No description available."))
        st.caption(f"{movie.get('year') or 'Year unknown'} · {', '.join(movie.get('genres', ['Other']))}")
        mid = str(movie["id"])
        score = st.slider("Your rating (1 = dislike, 5 = love)", 1, 5, st.session_state.ratings.get(mid, 3), key=f"slider_{mid}")
        if st.button("Save rating", type="primary"):
            st.session_state.ratings[mid] = int(score)
            save_data()
            st.success(f"Saved {score}/5 for {movie['title']}.")
            st.rerun()
        if mid in st.session_state.ratings and st.button("Remove my rating", key=f"remove_{mid}"):
            del st.session_state.ratings[mid]
            save_data()
            st.rerun()

else:
    st.markdown("<div class='hero'><div class='eyebrow'>Your activity</div><h1>📊 My Ratings</h1><p>Review the ratings you have saved.</p></div>", unsafe_allow_html=True)
    catalog = make_catalog(st.session_state.online_movies)
    lookup = {str(m["id"]): m for m in catalog}
    rows = []
    for mid, score in st.session_state.ratings.items():
        movie = lookup.get(mid)
        if movie:
            rows.append({"Movie": movie["title"], "Your rating": score, "Year": movie.get("year") or "Unknown", "Genres": ", ".join(movie.get("genres", ["Other"]))})
        else:
            rows.append({"Movie": f"Saved movie ({mid})", "Your rating": score, "Year": "Unknown", "Genres": "Details not loaded this session"})
    if not rows:
        st.info("You have not rated any movies yet. Open Rate Movies to get started.")
    else:
        rows.sort(key=lambda r: r["Your rating"], reverse=True)
        st.dataframe(rows, use_container_width=True, hide_index=True)
        a, b = st.columns(2)
        a.metric("Movies rated", len(rows))
        b.metric("Average rating", f"{sum(r['Your rating'] for r in rows)/len(rows):.1f}/5")
        st.bar_chart({r["Movie"]: r["Your rating"] for r in rows})

st.divider()
st.caption("CineMatch · Favorites, ratings and watchlist are stored locally in data/user_data.json. Online movie search uses Cinemeta only and does not require an API key. Search providers may occasionally be unavailable.")
