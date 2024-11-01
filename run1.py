import pandas as pd
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from flask import Flask, jsonify
import requests
from datetime import datetime

app = Flask(__name__)

# 서버의 API 엔드포인트 설정
USER_API_URL = 'http://localhost:8080/api/users'
RESTAURANT_API_URL = 'http://localhost:8080/api/restaurants/all'
REVIEW_LOG_API_URL = 'http://localhost:8080/api/review-logs'

print("데이터를 로드하고 추천 결과를 계산합니다. 잠시만 기다려주세요...")

# 사용자 데이터 가져오기
user_response = requests.get(USER_API_URL)
if user_response.status_code != 200:
    raise Exception('Failed to load user data')
user_data = user_response.json()
user_ids = [user['id'] for user in user_data]

# 레스토랑 데이터 가져오기
restaurant_response = requests.get(RESTAURANT_API_URL)
if restaurant_response.status_code != 200:
    raise Exception('Failed to load restaurant data')
api_data = restaurant_response.json()
restaurant_ids = [restaurant['id'] for restaurant in api_data]

# 유저-레스토랑 평점 매트릭스 생성 (초기값은 NaN으로 설정)
df = pd.DataFrame(index=user_ids, columns=restaurant_ids)

# 리뷰 로그 데이터 가져오기
review_response = requests.get(REVIEW_LOG_API_URL)
if review_response.status_code != 200:
    raise Exception('Failed to load review data')
review_data = review_response.json()

# 리뷰 데이터프레임 생성
review_df = pd.DataFrame([{
    '리뷰 ID': review['id'],
    '작성일시': review['createdAt'],
    '평점': review['rating'],
    '리뷰 내용': review['content'],
    '레스토랑 ID': review['restaurantId'],
    '작성자 ID': review['author']['id'],
    '작성자 이름': review['author']['username'],
    '작성자 이메일': review['author']['email']
} for review in review_data])

review_df['작성일시'] = pd.to_datetime(review_df['작성일시'])

user_total = review_df.groupby('작성자 ID')['평점'].agg(['sum', 'count']).rename(
    columns={'sum': 'user_total_rating', 'count': 'user_total_count'})

user_restaurant_total = review_df.groupby(['작성자 ID', '레스토랑 ID'])['평점'].agg(['sum', 'count']).rename(
    columns={'sum': 'user_restaurant_rating', 'count': 'user_restaurant_count'})

review_df = review_df.merge(user_total, on='작성자 ID')
review_df = review_df.merge(user_restaurant_total, on=['작성자 ID', '레스토랑 ID'])

def calculate_user_other_restaurants_avg(row):
    total_rating = row['user_total_rating'] - row['user_restaurant_rating']
    total_count = row['user_total_count'] - row['user_restaurant_count']
    return total_rating / total_count if total_count > 0 else np.nan

review_df['유저_다른_레스토랑_평균'] = review_df.apply(calculate_user_other_restaurants_avg, axis=1)

user_restaurant_max = review_df.groupby(['작성자 ID', '레스토랑 ID'])['평점'].max().rename('유저_레스토랑_최고_평점')
review_df = review_df.merge(user_restaurant_max, on=['작성자 ID', '레스토랑 ID'])

user_restaurant_latest = review_df.groupby(['작성자 ID', '레스토랑 ID'])['작성일시'].max().rename('유저_레스토랑_최근_리뷰일')
review_df = review_df.merge(user_restaurant_latest, on=['작성자 ID', '레스토랑 ID'])

current_time = datetime.now()
review_df['유저_레스토랑_최근_리뷰_경과시간(일)'] = (current_time - review_df['유저_레스토랑_최근_리뷰일']).dt.total_seconds() / (60 * 60 * 24)
review_df['유저_레스토랑_리뷰_개수'] = review_df['user_restaurant_count']

review_df = review_df.sort_values('작성일시', ascending=False).drop_duplicates(subset=['작성자 ID', '레스토랑 ID'])

for _, row in review_df.iterrows():
    user_id = row['작성자 ID']
    restaurant_id = row['레스토랑 ID']
    rating = row['평점']
    if user_id in df.index and restaurant_id in df.columns:
        df.at[user_id, restaurant_id] = rating

api_df = pd.DataFrame(api_data)

numeric_features_restaurant = api_df[['rating', 'userRating']].fillna(0)
scaler = StandardScaler()
numeric_scaled_restaurant = scaler.fit_transform(numeric_features_restaurant)
api_df['categories_text'] = api_df['categories'].apply(lambda x: ' '.join([cat['name'] for cat in x]))
api_df['menu_text'] = api_df['menu'].apply(lambda x: ' '.join(x))
api_df['text_features'] = api_df['categories_text'] + ' ' + api_df['menu_text']

vectorizer = TfidfVectorizer()
text_features_restaurant = vectorizer.fit_transform(api_df['text_features'])
combined_features_restaurant = hstack([numeric_scaled_restaurant, text_features_restaurant])
cosine_sim = cosine_similarity(combined_features_restaurant)
cosine_sim_df_item = pd.DataFrame(cosine_sim, index=api_df['id'], columns=api_df['id'])

# 사용자 기반 협업 필터링을 위한 사용자 유사도 매트릭스 계산
user_text = review_df.groupby('작성자 ID')['리뷰 내용'].apply(' '.join).reset_index()
numeric_features = [
    'user_total_rating', 'user_total_count', '유저_다른_레스토랑_평균',
    '유저_레스토랑_최고_평점', '유저_레스토랑_최근_리뷰_경과시간(일)', '유저_레스토랑_리뷰_개수'
]
user_numeric = review_df.groupby('작성자 ID')[numeric_features].mean().reset_index().fillna(0)
user_features = pd.merge(user_text, user_numeric, on='작성자 ID', how='outer').fillna('')
text_features = vectorizer.fit_transform(user_features['리뷰 내용'])
numeric_data = scaler.fit_transform(user_features[numeric_features])
numeric_sparse = csr_matrix(numeric_data)
combined_features_user = hstack([numeric_sparse, text_features])
cosine_sim_user = cosine_similarity(combined_features_user)
cosine_sim_df_user = pd.DataFrame(cosine_sim_user, index=user_features['작성자 ID'], columns=user_features['작성자 ID'])

# 사용자 기반 협업 필터링을 통한 예측 함수
def predict_ratings_user_based(df, sim_df_user):
    pred = df.copy()
    for user in df.index:
        for item in df.columns:
            if pd.isna(df.at[user, item]):
                similar_users = sim_df_user[user].drop(user).sort_values(ascending=False)
                item_ratings = df[item].dropna()
                similar_users = similar_users.loc[item_ratings.index]
                if not similar_users.empty:
                    weighted_sum = np.dot(similar_users, item_ratings.loc[similar_users.index])
                    sim_sum = similar_users.sum()
                    pred.at[user, item] = weighted_sum / sim_sum if sim_sum != 0 else np.nan
    return pred

predicted_df_user_based = predict_ratings_user_based(df, cosine_sim_df_user)

# 아이템 기반 협업 필터링을 통한 예측 함수
def predict_ratings_item_based(df, sim_df):
    pred = df.copy()
    for user in df.index:
        for restaurant in df.columns:
            if pd.isna(df.at[user, restaurant]):
                if restaurant in sim_df.columns:
                    similar_restaurants = sim_df[restaurant].drop(restaurant).sort_values(ascending=False)
                    user_ratings = df.loc[user, similar_restaurants.index].dropna()
    
                    if not user_ratings.empty:
                        similarities = similar_restaurants.loc[user_ratings.index]
                        weighted_sum = np.dot(user_ratings, similarities)
                        sim_sum = similarities.sum()
                        pred.at[user, restaurant] = weighted_sum / sim_sum if sim_sum != 0 else np.nan
                    else:
                        pred.at[user, restaurant] = np.nan
    return pred

# 예측 평점 계산 (아이템 기반)
predicted_df_item_based = predict_ratings_item_based(df, cosine_sim_df_item)

# 하이브리드 예측 함수
def hybrid_predict(predicted_df_user_based, predicted_df_item_based):
    hybrid_pred = predicted_df_user_based.combine_first(predicted_df_item_based)
    both_non_nan = predicted_df_user_based.notna() & predicted_df_item_based.notna()
    hybrid_pred[both_non_nan] = (predicted_df_user_based[both_non_nan] + predicted_df_item_based[both_non_nan]) / 2
    return hybrid_pred

# 하이브리드 예측 평점 계산
hybrid_predicted_df = hybrid_predict(predicted_df_user_based, predicted_df_item_based)

print("추천 결과 계산이 완료되었습니다.")

@app.route('/recommendations', methods=['GET'])
def get_recommendations():
    recommendations_json = hybrid_predicted_df.to_json(orient='index')
    return jsonify(json.loads(recommendations_json))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
