"""
Music Recommendation System — Rhombix Technologies Internship Task 1
Hybrid approach: Collaborative Filtering + Content-Based Filtering
Dataset: Spotify-style (user_id, song_id, repeated_play label, timestamps)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n_users=200, n_songs=100, seed=42):
    """Generate a Spotify-style dataset with play history and song features."""
    np.random.seed(seed)

    # User-song interaction log (mimics Spotify listening history)
    n_interactions = 3000
    user_ids  = np.random.randint(1, n_users + 1, n_interactions)
    song_ids  = np.random.randint(1, n_songs  + 1, n_interactions)
    # repeated_play: 1 if user replayed song within a month
    repeated  = np.random.choice([0, 1], n_interactions, p=[0.45, 0.55])
    timestamps = pd.date_range("2023-01-01", periods=n_interactions, freq="1h")

    interactions = pd.DataFrame({
        "user_id":       user_ids,
        "song_id":       song_ids,
        "repeated_play": repeated,
        "timestamp":     timestamps
    }).drop_duplicates(subset=["user_id", "song_id"])

    # Song metadata (content features)
    songs = pd.DataFrame({
        "song_id":    range(1, n_songs + 1),
        "title":      [f"Song_{i}"   for i in range(1, n_songs + 1)],
        "artist":     [f"Artist_{np.random.randint(1,20)}" for _ in range(n_songs)],
        "genre":      np.random.choice(["Pop","Rock","Hip-Hop","Jazz","Electronic","R&B"], n_songs),
        "tempo":      np.random.uniform(60, 180, n_songs),       # BPM
        "energy":     np.random.uniform(0, 1, n_songs),
        "valence":    np.random.uniform(0, 1, n_songs),          # positivity
        "danceability": np.random.uniform(0, 1, n_songs),
        "acousticness": np.random.uniform(0, 1, n_songs),
    })

    return interactions, songs


# ─────────────────────────────────────────────
# 2. COLLABORATIVE FILTERING (SVD-based)
# ─────────────────────────────────────────────

class CollaborativeFilter:
    def __init__(self, n_factors=20):
        self.n_factors = n_factors

    def fit(self, interactions: pd.DataFrame):
        # Build user-song matrix; value = repeated_play score
        self.user_index = {u: i for i, u in enumerate(interactions["user_id"].unique())}
        self.song_index = {s: i for i, s in enumerate(interactions["song_id"].unique())}
        self.idx_to_song = {i: s for s, i in self.song_index.items()}

        rows = interactions["user_id"].map(self.user_index)
        cols = interactions["song_id"].map(self.song_index)
        vals = interactions["repeated_play"].astype(float)

        self.matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_index), len(self.song_index))
        ).toarray()

        # Mean-center & SVD
        self.user_means = self.matrix.mean(axis=1, keepdims=True)
        centered = self.matrix - self.user_means
        k = min(self.n_factors, min(centered.shape) - 1)
        U, sigma, Vt = svds(centered, k=k)
        self.predicted = U @ np.diag(sigma) @ Vt + self.user_means

    def recommend(self, user_id: int, top_n=10, exclude_seen=True) -> list:
        if user_id not in self.user_index:
            return []
        u_idx = self.user_index[user_id]
        scores = self.predicted[u_idx]

        if exclude_seen:
            seen_mask = self.matrix[u_idx] > 0
            scores = np.where(seen_mask, -np.inf, scores)

        top_indices = np.argsort(scores)[::-1][:top_n]
        return [self.idx_to_song[i] for i in top_indices if i in self.idx_to_song]


# ─────────────────────────────────────────────
# 3. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

class ContentBasedFilter:
    def __init__(self):
        self.feature_cols = ["tempo", "energy", "valence", "danceability", "acousticness"]

    def fit(self, songs: pd.DataFrame):
        self.songs = songs.set_index("song_id")
        scaler = MinMaxScaler()
        self.feature_matrix = scaler.fit_transform(self.songs[self.feature_cols])
        # One-hot encode genre
        genre_dummies = pd.get_dummies(self.songs["genre"]).values
        self.feature_matrix = np.hstack([self.feature_matrix, genre_dummies])
        self.similarity = cosine_similarity(self.feature_matrix)
        self.song_id_list = self.songs.index.tolist()
        self.song_to_idx  = {s: i for i, s in enumerate(self.song_id_list)}

    def recommend(self, liked_songs: list, top_n=10) -> list:
        if not liked_songs:
            return []
        indices = [self.song_to_idx[s] for s in liked_songs if s in self.song_to_idx]
        if not indices:
            return []
        avg_sim = self.similarity[indices].mean(axis=0)
        # Zero out already-liked
        for i in indices:
            avg_sim[i] = -1
        top_indices = np.argsort(avg_sim)[::-1][:top_n]
        return [self.song_id_list[i] for i in top_indices]


# ─────────────────────────────────────────────
# 4. HYBRID RECOMMENDER
# ─────────────────────────────────────────────

class HybridRecommender:
    def __init__(self, cf_weight=0.6, cb_weight=0.4):
        self.cf = CollaborativeFilter()
        self.cb = ContentBasedFilter()
        self.cf_weight = cf_weight
        self.cb_weight = cb_weight

    def fit(self, interactions: pd.DataFrame, songs: pd.DataFrame):
        self.interactions = interactions
        self.songs = songs
        self.cf.fit(interactions)
        self.cb.fit(songs)

    def _get_liked_songs(self, user_id: int) -> list:
        user_data = self.interactions[
            (self.interactions["user_id"] == user_id) &
            (self.interactions["repeated_play"] == 1)
        ]
        return user_data["song_id"].tolist()

    def recommend(self, user_id: int, top_n=10) -> pd.DataFrame:
        cf_recs = self.cf.recommend(user_id, top_n=top_n * 2)
        liked   = self._get_liked_songs(user_id)
        cb_recs = self.cb.recommend(liked, top_n=top_n * 2)

        all_songs = set(cf_recs + cb_recs)
        scores = {}
        for song in all_songs:
            cf_score = (top_n * 2 - cf_recs.index(song)) / (top_n * 2) if song in cf_recs else 0
            cb_score = (top_n * 2 - cb_recs.index(song)) / (top_n * 2) if song in cb_recs else 0
            scores[song] = self.cf_weight * cf_score + self.cb_weight * cb_score

        top_songs = sorted(scores, key=scores.get, reverse=True)[:top_n]

        results = self.songs[self.songs["song_id"].isin(top_songs)].copy()
        results["hybrid_score"] = results["song_id"].map(scores).round(4)
        results = results.sort_values("hybrid_score", ascending=False)
        return results[["song_id", "title", "artist", "genre", "hybrid_score"]].reset_index(drop=True)


# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

def evaluate(model: HybridRecommender, test_interactions: pd.DataFrame, top_n=10):
    """Precision@K and Recall@K over test set."""
    precisions, recalls = [], []
    for user_id in test_interactions["user_id"].unique():
        actual = set(test_interactions[
            (test_interactions["user_id"] == user_id) &
            (test_interactions["repeated_play"] == 1)
        ]["song_id"])
        if not actual:
            continue
        recs = model.recommend(user_id, top_n=top_n)
        predicted = set(recs["song_id"])
        hit = len(predicted & actual)
        precisions.append(hit / top_n)
        recalls.append(hit / len(actual))

    return {
        f"Precision@{top_n}": round(np.mean(precisions), 4),
        f"Recall@{top_n}":    round(np.mean(recalls),    4),
    }


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Hybrid Music Recommendation System")
    print("  Rhombix Technologies — ML Internship Task 1")
    print("=" * 55)

    # Generate data
    interactions, songs = generate_dataset(n_users=200, n_songs=100)
    print(f"\n[Dataset]  {len(interactions)} interactions | {len(songs)} songs | {interactions['user_id'].nunique()} users")

    # Train/test split by timestamp
    interactions = interactions.sort_values("timestamp")
    split = int(len(interactions) * 0.8)
    train, test = interactions.iloc[:split], interactions.iloc[split:]

    # Train
    model = HybridRecommender(cf_weight=0.6, cb_weight=0.4)
    model.fit(train, songs)
    print("[Model]    Hybrid (SVD Collaborative + Cosine Content-Based)")
    print(f"           CF weight=0.6 | CB weight=0.4\n")

    # Recommend for a sample user
    sample_user = train["user_id"].value_counts().idxmax()
    recs = model.recommend(sample_user, top_n=10)
    print(f"Top-10 Recommendations for User {sample_user}:")
    print(recs.to_string(index=False))

    # Evaluate
    metrics = evaluate(model, test, top_n=10)
    print(f"\n[Evaluation on held-out test set]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n[Done] Model ready for deployment.")