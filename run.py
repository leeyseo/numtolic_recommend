import pandas as pd
import json
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# user.json 파일 읽기
with open('./user.json', 'r', encoding='utf-8') as file:
    user_data = json.load(file)
    user_ids = [user['userId'] for user in user_data]

# api.json 파일 읽기
with open('./api.json', 'r', encoding='utf-8') as file:
    api_data = json.load(file)
    restaurant_ids = [restaurant['id'] for restaurant in api_data['content']]

# 유저-레스토랑 평점 매트릭스 생성 (초기값은 NaN으로 설정)
df = pd.DataFrame(index=user_ids, columns=restaurant_ids)


# JSON 파일 읽기
with open('./reviewlog.json', 'r', encoding='utf-8') as file:
    review_data = json.load(file)

# 데이터프레임 생성
# 중첩된 구조에서 필요한 필드를 분리하여 DataFrame으로 만듦
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

# 작성일시를 datetime 형식으로 변환
review_df['작성일시'] = pd.to_datetime(review_df['작성일시'])
# 유저별 전체 평점 합계와 개수 계산
user_total = review_df.groupby('작성자 ID')['평점'].agg(['sum', 'count']).rename(columns={'sum': 'user_total_rating', 'count': 'user_total_count'})

# 유저별 레스토랑별 평점 합계와 개수 계산
user_restaurant_total = review_df.groupby(['작성자 ID', '레스토랑 ID'])['평점'].agg(['sum', 'count']).rename(columns={'sum': 'user_restaurant_rating', 'count': 'user_restaurant_count'})

# review_df에 merge
review_df = review_df.merge(user_total, on='작성자 ID')
review_df = review_df.merge(user_restaurant_total, on=['작성자 ID', '레스토랑 ID'])

# 이 유저의 다른 레스토랑에 매기는 리뷰 평점의 평균 계산
def calculate_user_other_restaurants_avg(row):
    total_rating = row['user_total_rating'] - row['user_restaurant_rating']
    total_count = row['user_total_count'] - row['user_restaurant_count']
    if total_count > 0:
        return total_rating / total_count
    else:
        return None  # 또는 np.nan

review_df['유저_다른_레스토랑_평균'] = review_df.apply(calculate_user_other_restaurants_avg, axis=1)
# 유저별 레스토랑별 최대 평점 계산
user_restaurant_max = review_df.groupby(['작성자 ID', '레스토랑 ID'])['평점'].max().rename('유저_레스토랑_최고_평점')

# review_df에 merge
review_df = review_df.merge(user_restaurant_max, on=['작성자 ID', '레스토랑 ID'])
from datetime import datetime

# 유저별 레스토랑별 가장 최근 리뷰 날짜 계산
user_restaurant_latest = review_df.groupby(['작성자 ID', '레스토랑 ID'])['작성일시'].max().rename('유저_레스토랑_최근_리뷰일')

# review_df에 merge
review_df = review_df.merge(user_restaurant_latest, on=['작성자 ID', '레스토랑 ID'])

# 현재 시간과의 차이 계산
current_time = datetime.now()
review_df['유저_레스토랑_최근_리뷰_경과시간(일)'] = (current_time - review_df['유저_레스토랑_최근_리뷰일']).dt.total_seconds() / (60 * 60 * 24)
# 이미 계산된 'user_restaurant_count' 컬럼 사용
review_df['유저_레스토랑_리뷰_개수'] = review_df['user_restaurant_count']
#작성자 ID와 레스토랑 ID를 기준으로 가장 최근의 행만 남김
review_df = review_df.sort_values('작성일시', ascending=False).drop_duplicates(subset=['작성자 ID', '레스토랑 ID'])


# 1. 사용자별 리뷰 내용을 하나의 문서로 결합
user_text = review_df.groupby('작성자 ID')['리뷰 내용'].apply(' '.join).reset_index()

# 2. 사용자별 수치형 특징 추출 (NaN 값을 0으로 대체)
numeric_features = [
    'user_total_rating', 'user_total_count', '유저_다른_레스토랑_평균',
    '유저_레스토랑_최고_평점', '유저_레스토랑_최근_리뷰_경과시간(일)', '유저_레스토랑_리뷰_개수'
]

user_numeric = review_df.groupby('작성자 ID')[numeric_features].mean().reset_index()
user_numeric = user_numeric.fillna(0)  # NaN 값을 0으로 대체

# 3. 텍스트 특징과 수치형 특징 결합
user_features = pd.merge(user_text, user_numeric, on='작성자 ID')

# 4. 텍스트 특징의 TF-IDF 벡터화
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(user_features['리뷰 내용'])

# 5. 수치형 특징 스케일링
scaler = StandardScaler()
numeric_data = scaler.fit_transform(user_features[numeric_features])



# 수치형 데이터를 희소 행렬로 변환
numeric_sparse = csr_matrix(numeric_data)

# 수치형 특징과 텍스트 특징을 결합
combined_features = hstack([numeric_sparse, text_features])

# 7. 코사인 유사도 계산
cosine_sim = cosine_similarity(combined_features)
cosine_sim_df_user = pd.DataFrame(cosine_sim, index=user_features['작성자 ID'], columns=user_features['작성자 ID'])
# index와 columns의 이름 제거
cosine_sim_df_user.index.name = None
cosine_sim_df_user.columns.name = None

# review_df의 작성자 ID와 레스토랑 ID를 기준으로 평점 값을 df에 입력
for _, row in review_df.iterrows():
    user_id = row['작성자 ID']
    restaurant_id = row['레스토랑 ID']
    rating = row['평점']
    
    # 해당 user_id와 restaurant_id가 df에 있는지 확인하고 평점 입력
    if user_id in df.index and restaurant_id in df.columns:
        df.at[user_id, restaurant_id] = rating

import pandas as pd
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# api.json 파일 읽기
with open('./api.json', 'r', encoding='utf-8') as file:
    api_content = json.load(file)['content']

# 데이터프레임 생성
api_df = pd.DataFrame(api_content)

# 수치형 특징 (rating, userRating)
numeric_features = api_df[['rating', 'userRating']].fillna(0)
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

# 텍스트 특징 (categories, menu)
api_df['categories_text'] = api_df['categories'].apply(lambda x: ' '.join([cat['name'] for cat in x]))
api_df['menu_text'] = api_df['menu'].apply(lambda x: ' '.join(x))
api_df['text_features'] = api_df['categories_text'] + ' ' + api_df['menu_text']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(api_df['text_features'])

# 특성 결합
combined_features = hstack([numeric_scaled, text_features])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(combined_features)
cosine_sim_df = pd.DataFrame(cosine_sim, index=api_df['id'], columns=api_df['id'])

# index와 columns의 이름 제거
cosine_sim_df.index.name = None
cosine_sim_df.columns.name = None


# api.json 파일 읽기
with open('./api.json', 'r', encoding='utf-8') as file:
    api_content = json.load(file)['content']

# 데이터프레임 생성
api_df = pd.DataFrame(api_content)

# 수치형 특징 (rating, userRating)
numeric_features = api_df[['rating', 'userRating']].fillna(0)
scaler = StandardScaler()
numeric_scaled = scaler.fit_transform(numeric_features)

# 텍스트 특징 (categories, menu)
api_df['categories_text'] = api_df['categories'].apply(lambda x: ' '.join([cat['name'] for cat in x]))
api_df['menu_text'] = api_df['menu'].apply(lambda x: ' '.join(x))
api_df['text_features'] = api_df['categories_text'] + ' ' + api_df['menu_text']

# TF-IDF 벡터화
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(api_df['text_features'])

# 특성 결합
combined_features = hstack([numeric_scaled, text_features])

# 코사인 유사도 계산
cosine_sim = cosine_similarity(combined_features)
cosine_sim_df_item = pd.DataFrame(cosine_sim, index=api_df['id'], columns=api_df['id'])

# index와 columns의 이름 제거
cosine_sim_df_item.index.name = None
cosine_sim_df_item.columns.name = None


def predict_ratings_item_based(df, sim_df):
    pred = df.copy()
    for user in df.index:
        for restaurant in df.columns:
            if pd.isna(df.at[user, restaurant]):
                # 유사한 레스토랑들 중 유저가 평가한 레스토랑 점수를 가져옴
                if restaurant in sim_df.columns:
                    similar_restaurants = sim_df[restaurant].drop(restaurant).sort_values(ascending=False)
                    user_ratings = df.loc[user, similar_restaurants.index].dropna()
    
                    if not user_ratings.empty:
                        # 유사도 가중치 적용하여 예측 평점 계산
                        similarities = similar_restaurants.loc[user_ratings.index]
                        weighted_sum = np.dot(user_ratings, similarities)
                        sim_sum = similarities.sum()
                        
                        # 예측 평점 계산
                        pred.at[user, restaurant] = weighted_sum / sim_sum if sim_sum != 0 else np.nan
                    else:
                        pred.at[user, restaurant] = np.nan
                else:
                    pred.at[user, restaurant] = np.nan
    return pred

# 예측 평점 계산
predicted_df_item_based = predict_ratings_item_based(df, cosine_sim_df)

predicted_df_item_based = predicted_df_item_based.fillna(0.0)


all_users = df.index  # 전체 사용자 목록은 df의 인덱스에서 가져옴

# 1. 사용자별 리뷰 내용을 하나의 문서로 결합
user_text = review_df.groupby('작성자 ID')['리뷰 내용'].apply(' '.join).reset_index()

# 2. 사용자별 수치형 특징 추출 (NaN 값을 0으로 대체)
numeric_features = [
    'user_total_rating', 'user_total_count', '유저_다른_레스토랑_평균',
    '유저_레스토랑_최고_평점', '유저_레스토랑_최근_리뷰_경과시간(일)', '유저_레스토랑_리뷰_개수'
]

user_numeric = review_df.groupby('작성자 ID')[numeric_features].mean().reset_index()
user_numeric = user_numeric.fillna(0)  # NaN 값을 0으로 대체

# 3. 텍스트 특징과 수치형 특징 결합 (모든 사용자 포함)
user_features = pd.merge(user_text, user_numeric, on='작성자 ID', how='outer').fillna('')

# 4. 텍스트 특징의 TF-IDF 벡터화
vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(user_features['리뷰 내용'])

# 5. 수치형 특징 스케일링
scaler = StandardScaler()
numeric_data = scaler.fit_transform(user_features[numeric_features])

# 6. 특성 결합 (수치형 + 텍스트)
numeric_sparse = csr_matrix(numeric_data)  # 희소 행렬로 변환
combined_features = hstack([numeric_sparse, text_features])

# 7. 코사인 유사도 계산
cosine_sim = cosine_similarity(combined_features)
cosine_sim_df_user = pd.DataFrame(cosine_sim, index=user_features['작성자 ID'], columns=user_features['작성자 ID'])

# 8. 전체 사용자 목록에 맞춰 유사도 행렬 확장 (df 인덱스 기준)
cosine_sim_df_user = cosine_sim_df_user.reindex(index=all_users, columns=all_users, fill_value=0)

# 3. 사용자 기반 협업 필터링을 통한 예측 함수 정의
def predict_ratings_user_based(df, cosine_sim_df_user):
    pred = df.copy()
    for user in df.index:
        for item in df.columns:
            if pd.isna(df.at[user, item]):
                # 유사한 유저들 중 해당 아이템에 평점을 매긴 유저의 평점과 유사도를 가져옴
                similar_users = cosine_sim_df_user[user].drop(user).sort_values(ascending=False)
                item_ratings = df[item].dropna()
                similar_users = similar_users.loc[item_ratings.index]
                
                if not similar_users.empty:
                    # 유사도 가중치 적용하여 예측 평점 계산
                    weighted_sum = np.dot(similar_users, item_ratings.loc[similar_users.index])
                    sim_sum = similar_users.sum()
                    pred.at[user, item] = weighted_sum / sim_sum if sim_sum != 0 else np.nan
                else:
                    pred.at[user, item] = np.nan
    return pred

# 4. 예측 평점 계산
predicted_df_user_based = predict_ratings_user_based(df, cosine_sim_df_user)


# 1. 아이템 기반 협업 필터링을 통한 예측 함수 (이미 존재한다고 가정)
def predict_ratings_item_based(df, item_sim_df):
    pred = df.copy()
    for user in df.index:
        for item in df.columns:
            if pd.isna(df.at[user, item]):
                similar_items = item_sim_df[item].drop(item).sort_values(ascending=False)
                user_ratings = df.loc[user, similar_items.index].dropna()
                
                if not user_ratings.empty:
                    similarities = similar_items.loc[user_ratings.index]
                    weighted_sum = np.dot(user_ratings, similarities)
                    sim_sum = similarities.sum()
                    pred.at[user, item] = weighted_sum / sim_sum if sim_sum != 0 else np.nan
                else:
                    pred.at[user, item] = np.nan
    return pred


# 수정된 하이브리드 예측 함수
def hybrid_predict(predicted_df_user_based, predicted_df_item_based):
    # 우선 NaN이 아닌 값을 우선적으로 사용하여 병합
    hybrid_pred = predicted_df_user_based.combine_first(predicted_df_item_based)
    
    # 둘 다 값이 존재하는 경우 평균으로 덮어쓰도록 설정
    both_non_nan = predicted_df_user_based.notna() & predicted_df_item_based.notna()
    hybrid_pred[both_non_nan] = (predicted_df_user_based[both_non_nan] + predicted_df_item_based[both_non_nan]) / 2
    
    return hybrid_pred

# 4. 하이브리드 예측 평점 계산
hybrid_predicted_df = hybrid_predict(predicted_df_user_based, predicted_df_item_based)


# hybrid_predicted_df를 CSV 파일로 저장
file_path = "hybrid_predicted_df.csv"
hybrid_predicted_df.to_csv(file_path, index=True)  # 인덱스 포함하여 저장

print(f"DataFrame이 로컬에 '{file_path}' 파일로 저장되었습니다.")