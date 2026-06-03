# Music Recommendation System

A hybrid music recommendation system inspired by Spotify's engine, combining **Collaborative Filtering** and **Content-Based Filtering** to predict songs a user is likely to replay within a month.

---

## How it works

| Component | Method | Weight |
|---|---|---|
| Collaborative Filtering | SVD (20 latent factors) | 60% |
| Content-Based Filtering | Cosine Similarity on audio features | 40% |

**Audio features used:** tempo, energy, valence, danceability, acousticness, genre

---

## Dataset

Spotify-style dataset with:
- 2,776 user-song interactions
- 200 users · 100 songs
- Binary label: `1` = song replayed within a month, `0` = not replayed
- Timestamps for train/test splitting

---

## Results

| Metric | Score |
|---|---|
| Precision@10 | 0.0235 |
| Recall@10 | 0.1279 |

---

## How to run

**1. Install dependencies**
```bash
pip install numpy pandas scikit-learn scipy
```

**2. Run the script**
```bash
python "Music recommendation.py"
```

---

## Sample output

```
Top-10 Recommendations for User 190:
 song_id   title    artist   genre  hybrid_score
      25 Song_25 Artist_16     R&B          0.72
      91 Song_91 Artist_15     Pop          0.57
      39 Song_39  Artist_9 Hip-Hop          0.54
      10 Song_10 Artist_17 Hip-Hop          0.51
      66 Song_66 Artist_18     R&B          0.50

Precision@10: 0.0235
Recall@10:    0.1279
```

---

## Dependencies

- Python 3.12+
- numpy
- pandas
- scikit-learn
- scipy
