# 🎬 CineMatch — Movie Recommendation System

CineMatch is a movie discovery and recommendation web app built with **Python** and **Streamlit**. It lets users search for movies through **Cinemeta**, browse movie details and posters when available, and keep track of personal favorites, watchlists, and ratings.

The project also includes a recommendation system that uses movie metadata to suggest similar titles. Movie data returned by Cinemeta depends on its catalogue and service availability.

---

## 🚀 Features

### 🔹 Movie Search & Discovery
- Search for movies using Cinemeta without needing a TMDB API key.
- Browse movie information such as title, release year, description, genres, and ratings when provided by the source.
- Display movie posters when an image URL is available.
- Use a built-in sample movie catalogue when exploring the app.
- Filter or browse movies by available genre information.

### 🔹 Recommendations
- Generate movie recommendations using movie metadata.
- Use TF-IDF vectorization and cosine similarity to find titles with similar descriptive features.
- Discover movies related to a selected title.

### 🔹 Personal Movie Library
- Mark movies as favorites.
- Add movies to a watchlist.
- Save personal movie ratings.
- Store user data locally in JSON files so it can persist between app runs.

### 🔹 Poster Handling
- The app attempts to display poster images supplied by Cinemeta.
- It can try an additional Cinemeta metadata lookup for movies with an IMDb ID.
- If Cinemeta has no image for a title, or its service/image host is unavailable, the placeholder may still appear.

---

## 📂 Folder Structure

```text
CineMatch/
├── app.py
├── data/
│   ├── user_data.json       # Local favorites, watchlist, and rating data (created/used by the app)
│   └── ratings.json         # Legacy ratings data, if present
└── README.md
```

The exact files inside `data/` may vary depending on the version of the app and whether the app has already been run. Do not delete this folder if you want to preserve saved user data.

---

## ⚙️ Tech Stack

**Frontend / Web App**
- Python
- Streamlit

**Movie Search & Data**
- Cinemeta API (no API key required)
- Requests / HTTP calls to retrieve online movie information

**Recommendation Engine**
- pandas
- scikit-learn
- TF-IDF vectorization
- Cosine similarity

**Local Storage**
- JSON files

---

## 🧑‍💻 Installation & Setup

### 1️⃣ Get the Project

If you have a GitHub repository, clone it:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd <YOUR-REPOSITORY-FOLDER>
```

Or open the project folder directly in VS Code. Replace the placeholders with your actual repository URL and folder name.

### 2️⃣ Create and Activate a Virtual Environment (Recommended)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

Install the packages used by the app:

```bash
python -m pip install streamlit requests pandas scikit-learn
```

If your project includes a `requirements.txt` file, you can install from it instead:

```bash
python -m pip install -r requirements.txt
```

### 4️⃣ Run CineMatch

From the directory containing `app.py`, run:

```bash
streamlit run app.py
```

Streamlit will print a local URL in the terminal, usually `http://localhost:8501`. Open that URL in your browser.

---

## 🖥️ How to Use

1. Start the app using `streamlit run app.py`.
2. Open the local URL shown in the terminal.
3. Use the search/discovery section to look up a movie.
4. Browse the movie information and poster if one is available.
5. Use the available genre or discovery filters to narrow down titles.
6. Add movies to **Favorites** or your **Watchlist**.
7. Rate movies to help maintain your personal movie history.
8. Explore recommendations for similar movies.

The exact labels and placement of controls may differ slightly between app versions.

---

## 🎞️ Movie Data & Poster Limitations

CineMatch uses **Cinemeta only** for online movie metadata and poster lookup. It does not use TMDB or IMDbOT.

- No API key or account is required for Cinemeta.
- Search results and available fields depend on Cinemeta's catalogue.
- Not every movie has a poster image in the returned metadata.
- If a poster is missing from Cinemeta, the app may show **“Poster unavailable”**.
- The extra metadata lookup can help for some movies with valid IMDb IDs, but it cannot guarantee a poster for every title.
- Online search requires an internet connection and a working Cinemeta service.

---

## 💾 Local User Data

Favorites, watchlist entries, and personal ratings are saved locally in JSON files under the `data/` directory.

- Keep the `data/` directory when replacing or updating `app.py`.
- Back up the JSON files before making major changes.
- Do not upload personal data files to a public repository if they contain information you want to keep private.

---

## 🛠️ Troubleshooting

### The app does not start
- Confirm Python is installed.
- Activate your virtual environment, if you created one.
- Install the required packages.
- Run the command from the folder containing `app.py`.

### Movie search returns no results
- Check your internet connection.
- Try another movie title.
- Cinemeta may be temporarily unavailable or may not list that title.

### Posters are missing
- Confirm the movie has a poster URL in the Cinemeta result or metadata.
- Try refreshing the app or searching again.
- Some posters are not available from Cinemeta, so a placeholder may remain.

### Favorites or watchlist seem empty
- Confirm you are running the app from the same project folder.
- Check that the `data/` directory and its JSON files have not been deleted or moved.

---

## 🔐 Notes

- Cinemeta is an external service; its availability and data coverage are outside this project's control.
- The recommendation system can only make useful suggestions when sufficient movie metadata is available.
- This project is intended as a learning/demo project. Check the applicable service terms before using it commercially.

---

## 🔮 Future Improvements

- Add more robust poster recovery and image-error handling.
- Add pagination or “load more” for large search results.
- Improve filtering by language, release year, and genre where metadata supports it.
- Add automated tests for search parsing and recommendation logic.
- Add a `requirements.txt` file and sample screenshots.
- Improve error messages when Cinemeta is unavailable.

---

## ✨ Project Information

- **Project:** CineMatch — Movie Recommendation System
- **Category:** Python, Streamlit, Movie Discovery & Recommendation
- **Version:** 1.0.0
- **Movie metadata provider:** Cinemeta
- **Devloper** Krutant Tale
