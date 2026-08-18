import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --------------------------------------------------------------
# 1. Load CSV files
# --------------------------------------------------------------
print("Loading movies.csv ...")
movies = pd.read_csv('movies.csv', sep=';', header=0, encoding='utf-8')
movies.columns = ['movie_id', 'title', 'genres']
print(f"Movies loaded: {movies.shape[0]} rows, columns: {movies.columns.tolist()}")

ratings = pd.read_csv('ratings.csv', sep=';', header=0, encoding='utf-8')
ratings.columns = ['user_id', 'movie_id', 'rating']
print(f"Ratings loaded: {ratings.shape[0]} rows, columns: {ratings.columns.tolist()}")

# Ensure numeric types
movies['movie_id'] = pd.to_numeric(movies['movie_id'], errors='coerce').astype('Int64')
movies = movies.dropna(subset=['movie_id', 'title'])
ratings['user_id'] = pd.to_numeric(ratings['user_id'], errors='coerce').astype('Int64')
ratings['movie_id'] = pd.to_numeric(ratings['movie_id'], errors='coerce').astype('Int64')
ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
ratings = ratings.dropna(subset=['user_id', 'movie_id', 'rating'])

print(f"After cleaning: {movies.shape[0]} movies, {ratings.shape[0]} ratings")

# --------------------------------------------------------------
# 2. Feature Engineering
# --------------------------------------------------------------

# Extract year from title
def extract_year(title):
    if not isinstance(title, str):
        return None
    match = re.search(r'\((\d{4})\)', title)
    return int(match.group(1)) if match else None

movies['year'] = movies['title'].apply(extract_year)
movies = movies.dropna(subset=['year']).copy()
movies['decade'] = (movies['year'] // 10) * 10

# Build genre list from the 'genres' column (pipe-separated)
all_genres = set()
for genres_str in movies['genres'].dropna():
    if genres_str and genres_str != '(no genres listed)':
        for g in genres_str.split('|'):
            g = g.strip()
            if g:
                all_genres.add(g)
genre_list = sorted(all_genres)
print(f"Discovered {len(genre_list)} genres: {genre_list}")

# Create binary genre vectors
def genres_to_vec(genres_str):
    vec = np.zeros(len(genre_list))
    if not genres_str or genres_str == '(no genres listed)':
        return vec
    for g in genres_str.split('|'):
        g = g.strip()
        if g in genre_list:
            vec[genre_list.index(g)] = 1
    return vec

movies['genre_vec'] = movies['genres'].apply(genres_to_vec)
genre_df = pd.DataFrame(movies['genre_vec'].tolist(), columns=genre_list, index=movies.index)

# Decade encoding
decade_encoder = LabelEncoder()
movies['decade_idx'] = decade_encoder.fit_transform(movies['decade'])

# --- Item features ---
item_features = movies[['movie_id']].copy()
item_features[genre_list] = genre_df
item_features['decade_idx'] = movies['decade_idx']

# --- User features ---
# Merge ratings with movie genres
merged = ratings.merge(movies[['movie_id', 'genre_vec']], on='movie_id')

# User stats
user_stats = ratings.groupby('user_id').agg(
    rating_count=('rating', 'count'),
    rating_ave=('rating', 'mean')
).reset_index()

# User genre averages
def user_genre_avg(user_ratings):
    total = np.zeros(len(genre_list))
    cnt = np.zeros(len(genre_list))
    for _, row in user_ratings.iterrows():
        vec = row['genre_vec']
        r = row['rating']
        total += vec * r
        cnt += vec
    avg = np.divide(total, cnt, out=np.zeros_like(total), where=cnt!=0)
    return avg

genre_avg_df = merged.groupby('user_id').apply(user_genre_avg).reset_index()
genre_avg_df = genre_avg_df.rename(columns={0: 'genre_avg'})
genre_avg_df[genre_list] = pd.DataFrame(genre_avg_df['genre_avg'].tolist(), index=genre_avg_df.index)

user_features = user_stats.merge(genre_avg_df[['user_id'] + genre_list], on='user_id')

# Rename genre columns so user-side (preference average) and item-side
# (genre indicator) don't collide when merged together below
user_genre_cols = [f'{g}_user' for g in genre_list]
item_genre_cols = [f'{g}_item' for g in genre_list]
user_features = user_features.rename(columns=dict(zip(genre_list, user_genre_cols)))
item_features = item_features.rename(columns=dict(zip(genre_list, item_genre_cols)))

# --- Training pairs ---
train_df = ratings.merge(user_features, on='user_id')
train_df = train_df.merge(item_features, on='movie_id')

# Extract arrays
user_cols = ['user_id', 'rating_count', 'rating_ave'] + user_genre_cols
item_cols = ['movie_id'] + item_genre_cols + ['decade_idx']

user_train = train_df[user_cols].values
item_train = train_df[item_cols].values
y_train = train_df[['rating']].values

X_user_ids = user_train[:, 0].astype(int)
X_user_content = user_train[:, 1:].astype(np.float32)
X_item_ids = item_train[:, 0].astype(int)
X_item_content = item_train[:, 1:].astype(np.float32)
y = y_train.astype(np.float32)

# --- Train/Test Split ---
(
    X_user_ids_train, X_user_ids_test,
    X_user_content_train, X_user_content_test,
    X_item_ids_train, X_item_ids_test,
    X_item_content_train, X_item_content_test,
    y_train, y_test
) = train_test_split(
    X_user_ids, X_user_content, X_item_ids, X_item_content, y,
    test_size=0.2, random_state=42
)

# --- Scaling ---
scaler_user_content = StandardScaler()
X_user_content_train = scaler_user_content.fit_transform(X_user_content_train)
X_user_content_test = scaler_user_content.transform(X_user_content_test)

scaler_item_content = StandardScaler()
X_item_content_train = scaler_item_content.fit_transform(X_item_content_train)
X_item_content_test = scaler_item_content.transform(X_item_content_test)

scaler_target = MinMaxScaler((-1, 1))
y_train_scaled = scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_target.transform(y_test.reshape(-1, 1)).flatten()

# --------------------------------------------------------------
# 3. Build the Model
# --------------------------------------------------------------

num_users = int(ratings['user_id'].max()) + 1
num_items = int(movies['movie_id'].max()) + 1
embedding_dim = 32

# Custom layer replacing the Lambda(l2_normalize) layers.
# Some TensorFlow/Keras builds (notably on macOS) fail to infer the output
# shape of a Lambda layer at predict() time, raising:
#   ValueError: as_list() is not defined on an unknown TensorShape
# A proper subclassed Layer with compute_output_shape avoids that entirely.
class L2Normalize(layers.Layer):
    def call(self, x):
        return tf.linalg.l2_normalize(x, axis=1)
    def compute_output_shape(self, input_shape):
        return input_shape

# User tower
user_id_input = layers.Input(shape=(1,), name='user_id')
user_id_embed = layers.Embedding(num_users, embedding_dim)(user_id_input)
user_id_embed = layers.Flatten()(user_id_embed)

user_content_input = layers.Input(shape=(X_user_content_train.shape[1],), name='user_content')
user_content_dense = layers.Dense(64, activation='relu')(user_content_input)

user_combined = layers.Concatenate()([user_id_embed, user_content_dense])
user_combined = layers.Dense(64, activation='relu')(user_combined)
user_combined = layers.Dense(embedding_dim, activation='relu')(user_combined)
user_combined = L2Normalize()(user_combined)

# Item tower
item_id_input = layers.Input(shape=(1,), name='item_id')
item_id_embed = layers.Embedding(num_items, embedding_dim)(item_id_input)
item_id_embed = layers.Flatten()(item_id_embed)

item_content_input = layers.Input(shape=(X_item_content_train.shape[1],), name='item_content')
item_content_dense = layers.Dense(64, activation='relu')(item_content_input)

item_combined = layers.Concatenate()([item_id_embed, item_content_dense])
item_combined = layers.Dense(64, activation='relu')(item_combined)
item_combined = layers.Dense(embedding_dim, activation='relu')(item_combined)
item_combined = L2Normalize()(item_combined)

# Dot product
dot = layers.Dot(axes=1)([user_combined, item_combined])

model = keras.Model(inputs=[user_id_input, user_content_input, item_id_input, item_content_input],
                    outputs=dot)
model.compile(optimizer='adam', loss='mse')
model.summary()

# --------------------------------------------------------------
# 4. Train
# --------------------------------------------------------------

history = model.fit(
    [X_user_ids_train, X_user_content_train,
     X_item_ids_train, X_item_content_train],
    y_train_scaled,
    epochs=5,        # Increase to 20 for better accuracy
    batch_size=256,
    validation_data=(
        [X_user_ids_test, X_user_content_test,
         X_item_ids_test, X_item_content_test],
        y_test_scaled
    ),
    verbose=1
)

# Evaluate
y_pred_scaled = model.predict([X_user_ids_test, X_user_content_test,
                               X_item_ids_test, X_item_content_test])
y_pred = scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
mse = np.mean((y_test.flatten() - y_pred)**2)
print(f"Test MSE: {mse:.4f}")

# --------------------------------------------------------------
# 5. Recommendation Function
# --------------------------------------------------------------

def recommend_for_user(user_id, top_k=10, exclude_rated=True):
    user_row = user_features[user_features['user_id'] == user_id]
    if len(user_row) == 0:
        print("New user: using average user profile.")
        user_content = np.mean(user_features[user_genre_cols + ['rating_count', 'rating_ave']].values, axis=0)
        user_content = user_content.reshape(1, -1)
        user_content = scaler_user_content.transform(user_content)
        user_id_input = np.array([[user_id]])
    else:
        user_content = user_row[['rating_count', 'rating_ave'] + user_genre_cols].values.astype(np.float32)
        user_content = scaler_user_content.transform(user_content)
        user_id_input = np.array([[user_id]])

    rated_movies = ratings[ratings['user_id'] == user_id]['movie_id'].values if exclude_rated else []
    item_ids = movies['movie_id'].values.astype(int)
    item_content = item_features[item_genre_cols + ['decade_idx']].values.astype(np.float32)
    item_content_scaled = scaler_item_content.transform(item_content)

    preds = model(
        [tf.constant(np.tile(user_id_input, (len(item_ids), 1))),
        tf.constant(np.tile(user_content, (len(item_ids), 1))),
        tf.constant(item_ids.reshape(-1, 1)),
        tf.constant(item_content_scaled)],
        training=False
    ).numpy()
    preds = scaler_target.inverse_transform(preds).flatten()

    sorted_indices = np.argsort(-preds)
    if exclude_rated:
        sorted_indices = [i for i in sorted_indices if item_ids[i] not in rated_movies]

    top_movie_ids = item_ids[sorted_indices[:top_k]]
    top_scores = preds[sorted_indices[:top_k]]

    recommendations = movies[movies['movie_id'].isin(top_movie_ids)][['movie_id', 'title']].copy()
    recommendations['score'] = recommendations['movie_id'].map(dict(zip(top_movie_ids, top_scores)))
    return recommendations.sort_values('score', ascending=False)

# Example
print("\nRecommendations for user 1:")
print(recommend_for_user(1, top_k=10))