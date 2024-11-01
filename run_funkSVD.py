import pandas as pd
import json
from datetime import datetime
import numpy as np
from xgboost import XGBRegressor

# JSON 파일 읽기
with open('./reviewlog.json', 'r', encoding='utf-8') as file:
    review_data = json.load(file)

with open('./api.json', 'r', encoding='utf-8') as file:
    api_data = json.load(file)

# user.json 파일 읽기
with open('./user.json', 'r', encoding='utf-8') as file:
    user_data = json.load(file)

# visit_data가 단일 객체가 아닌 리스트인 경우를 대비
if isinstance(review_data, dict):
    review_data = [review_data]

# 필요한 데이터를 추출하기 위해 딕셔너리 형태로 변환 (restaurantId를 키로 설정)
api_content = {item['id']: item for item in api_data['content']}

# 데이터프레임 생성 준비
final_data = []

for review in review_data:
    rest_id = review['restaurantId']
    
    # 해당 restaurantId에 대한 apilog 데이터 가져오기
    if rest_id in api_content:
        api_entry = api_content[rest_id]
        
        # 필요한 데이터 추출
        user_rating = api_entry.get('userRating')
        travel_time = api_entry.get('travelTime')
        rating = api_entry.get('rating')
        
        # 메뉴 최대 3개 추출
        menu_items = api_entry.get('menu', [])
        menu1 = menu_items[0] if len(menu_items) > 0 else None
        menu2 = menu_items[1] if len(menu_items) > 1 else None
        menu3 = menu_items[2] if len(menu_items) > 2 else None
        
        # 카테고리 최대 2개 id 추출
        category_ids = [cat['id'] for cat in api_entry.get('categories', [])]
        category_id1 = category_ids[0] if len(category_ids) > 0 else None
        category_id2 = category_ids[1] if len(category_ids) > 1 else None
        
        # visit 정보와 결합
        final_data.append({
            '작성자 ID': review['author']['id'],
            '레스토랑 ID': rest_id,
            '작성일시': review['createdAt'],
            'reviewRating': review['rating'],
            'reviewText': review['content'],
            'userRating': user_rating,
            'travelTime': travel_time,
            'rating': rating,
            'store_menu1': menu1,
            'store_menu2': menu2,
            'store_menu3': menu3,
            'food_category_id1': category_id1,
            'food_category_id2': category_id2
        })

# 데이터프레임 생성
review_df = pd.DataFrame(final_data)

# 유저별 전체 평점 합계와 개수 계산
user_total = review_df.groupby('작성자 ID')['reviewRating'].agg(['sum', 'count']).rename(columns={'sum': 'user_total_rating', 'count': 'user_total_count'})

# 유저별 레스토랑별 평점 합계와 개수 계산
user_restaurant_total = review_df.groupby(['작성자 ID', '레스토랑 ID'])['reviewRating'].agg(['sum', 'count']).rename(columns={'sum': 'user_restaurant_rating', 'count': 'user_restaurant_count'})

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
user_restaurant_max = review_df.groupby(['작성자 ID', '레스토랑 ID'])['reviewRating'].max().rename('유저_레스토랑_최고_평점')

# review_df에 merge
review_df = review_df.merge(user_restaurant_max, on=['작성자 ID', '레스토랑 ID'])


# 유저별 레스토랑별 가장 최근 리뷰 날짜 계산
user_restaurant_latest = review_df.groupby(['작성자 ID', '레스토랑 ID'])['작성일시'].max().rename('유저_레스토랑_최근_리뷰일')

# review_df에 merge
review_df = review_df.merge(user_restaurant_latest, on=['작성자 ID', '레스토랑 ID'])

# '유저_레스토랑_최근_리뷰일'을 datetime 형식으로 변환
review_df['유저_레스토랑_최근_리뷰일'] = pd.to_datetime(review_df['유저_레스토랑_최근_리뷰일'])

# 현재 시간과의 차이 계산
current_time = datetime.now()
review_df['유저_레스토랑_최근_리뷰_경과시간(일)'] = (current_time - review_df['유저_레스토랑_최근_리뷰일']).dt.total_seconds() / (60 * 60 * 24)

# user.json 파일 읽기
with open('./user.json', 'r', encoding='utf-8') as file:
    user_data = json.load(file)
    user_ids = [user['userId'] for user in user_data]

# user_ids에 있는 아이디가 review_df에 없다면 결측치 행을 추가
missing_user_ids = [user_id for user_id in user_ids if user_id not in review_df['작성자 ID'].unique()]

# 결측치 행을 추가하기 위한 DataFrame 생성
missing_rows = pd.DataFrame({
    '작성자 ID': missing_user_ids,
    **{col: np.nan for col in review_df.columns if col != '작성자 ID'}
})

# review_df에 결측치 행 추가 및 작성자 ID로 정렬
review_df = pd.concat([review_df, missing_rows], ignore_index=True)
review_df = review_df.sort_values(by='작성자 ID').reset_index(drop=True)


user_ids = [user['userId'] for user in user_data]


# 'content' 키를 통해 레스토랑 리스트에 접근
restaurant_ids = [restaurant['id'] for restaurant in api_data['content']]

# 유저-아이템 모든 조합 생성
all_combinations = pd.MultiIndex.from_product([user_ids, restaurant_ids], names=['작성자 ID', '레스토랑 ID']).to_frame(index=False)

# 기존 review_df를 모든 조합에 맞춰 결합하고, 없는 조합은 결측치로 채우기
expanded_review_df = pd.merge(all_combinations, review_df, on=['작성자 ID', '레스토랑 ID'], how='left')

# 결과 출력
review_df = expanded_review_df.sort_values(by=['작성자 ID', '레스토랑 ID']).reset_index(drop=True)



# 유저-레스토랑 평점 매트릭스 생성 (초기값은 NaN으로 설정)
df = pd.DataFrame(index=user_ids, columns=restaurant_ids)

# review_df의 작성자 ID와 레스토랑 ID를 기준으로 평점 값을 df에 입력
for _, row in review_df.iterrows():
    user_id = row['작성자 ID']
    restaurant_id = row['레스토랑 ID']
    rating = row['rating']
    
    # 해당 user_id와 restaurant_id가 df에 있는지 확인하고 평점 입력
    if user_id in df.index and restaurant_id in df.columns:
        df.at[user_id, restaurant_id] = rating


# 설정: 잠재 요인 수, 초기 학습률, 정규화 파라미터, 반복 횟수
latent_features = 10
initial_learning_rate = 0.01
regularization = 0.1
iterations = 10000

# 유저-아이템 평점 행렬 R 생성
R = df.values  # 유저-아이템 평점 행렬 (결측값 포함)
num_users, num_items = R.shape

# 유저와 아이템 잠재 요인 행렬 초기화
P = np.random.normal(scale=1./latent_features, size=(num_users, latent_features))
Q = np.random.normal(scale=1./latent_features, size=(num_items, latent_features))

# FunkSVD 학습 과정
learning_rate = initial_learning_rate
for iteration in range(iterations):
    for i in range(num_users):
        for j in range(num_items):
            if not np.isnan(R[i, j]):  # 실제 평점이 있는 경우에만 업데이트
                # 예측 평점 및 오류 계산
                prediction = np.dot(P[i, :], Q[j, :].T)
                error = R[i, j] - prediction
                
                # 유저 및 아이템 잠재 요인 업데이트 (정규화 포함)
                P[i, :] += learning_rate * (error * Q[j, :] - regularization * P[i, :])
                Q[j, :] += learning_rate * (error * P[i, :] - regularization * Q[j, :])
                
    # 매 반복 후 손실 계산
    if (iteration + 1) % 10 == 0:
        # 손실 계산 (평점 예측 오류와 정규화 항)
        loss = 0
        for i in range(num_users):
            for j in range(num_items):
                if not np.isnan(R[i, j]):
                    prediction = np.dot(P[i, :], Q[j, :].T)
                    loss += (R[i, j] - prediction) ** 2 + regularization * (np.linalg.norm(P[i, :]) + np.linalg.norm(Q[j, :]))
        #print(f"Iteration: {iteration + 1}, Loss: {loss:.4f}")
    
    # 학습률 점진적 감소
    learning_rate *= 0.99

# 예측 평점 행렬 계산
predicted_ratings = np.dot(P, Q.T)

# DataFrame으로 변환하여 결측값 채운 결과 확인
predicted_df = pd.DataFrame(predicted_ratings, index=df.index, columns=df.columns)


# 1. reviewRating이 있는 데이터만 추출하여 학습 데이터 생성
train_df = review_df[review_df['reviewRating'].notna()]

# 2. FunkSVD로 생성한 예측 평점을 train_df에 추가
train_df['svd_pred'] = train_df.apply(lambda row: predicted_df.at[row['작성자 ID'], row['레스토랑 ID']], axis=1)

# 3. 학습을 위한 특성(X)과 타겟(y) 정의
X = train_df[['svd_pred', 'userRating', 'travelTime', 'rating', 'user_total_rating', 'user_total_count',
              'user_restaurant_rating', 'user_restaurant_count', '유저_다른_레스토랑_평균',
              '유저_레스토랑_최고_평점', '유저_레스토랑_최근_리뷰_경과시간(일)']].fillna(0)
y = train_df['reviewRating']

# 4. XGBoost 모델 학습
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1,
                         max_depth=5, random_state=42)
xgb_model.fit(X, y)

# 5. 모든 유저-아이템 조합에 대해 최종 예측 평점 생성
user_ids = review_df['작성자 ID'].unique()
item_ids = review_df['레스토랑 ID'].unique()
final_predicted_df = pd.DataFrame(index=user_ids, columns=item_ids)

# 6. 유저-아이템 조합에 대해 최종 평점 생성
for user_id in user_ids:
    for item_id in item_ids:
        if pd.isna(user_id) or pd.isna(item_id):
            continue  # user_id 또는 item_id가 NaN이면 건너뜁니다.

        # 유저가 해당 아이템에 리뷰를 남겼는지 확인
        user_item_reviews = train_df[(train_df['작성자 ID'] == user_id) & (train_df['레스토랑 ID'] == item_id)]

        if not user_item_reviews.empty:
            # 리뷰가 있는 경우, 가장 최근의 reviewRating 사용
            latest_review = user_item_reviews.sort_values(by='작성일시', ascending=False).iloc[0]
            final_predicted_df.at[user_id, item_id] = latest_review['reviewRating']
        else:
            if user_id in train_df['작성자 ID'].values:
                # 유저가 train_df에 있지만 해당 아이템에 리뷰가 없는 경우, XGBoost로 예측
                svd_pred = predicted_df.at[user_id, item_id] if pd.notna(predicted_df.at[user_id, item_id]) else 0

                # 유저 특성 가져오기
                user_features = train_df[train_df['작성자 ID'] == user_id].iloc[0]
                # 아이템 특성 가져오기
                item_features = review_df[review_df['레스토랑 ID'] == item_id].iloc[0]

                # XGBoost 입력 데이터 생성
                xgb_input = pd.DataFrame([{
                    'svd_pred': svd_pred,
                    'userRating': user_features['userRating'],
                    'travelTime': user_features['travelTime'],
                    'rating': item_features['rating'],
                    'user_total_rating': user_features['user_total_rating'],
                    'user_total_count': user_features['user_total_count'],
                    'user_restaurant_rating': user_features['user_restaurant_rating'],
                    'user_restaurant_count': user_features['user_restaurant_count'],
                    '유저_다른_레스토랑_평균': user_features['유저_다른_레스토랑_평균'],
                    '유저_레스토랑_최고_평점': user_features['유저_레스토랑_최고_평점'],
                    '유저_레스토랑_최근_리뷰_경과시간(일)': user_features['유저_레스토랑_최근_리뷰_경과시간(일)']
                }]).fillna(0)  # 결측값을 0으로 채웁니다.

                # XGBoost로 평점 예측
                pred_rating = xgb_model.predict(xgb_input)[0]
                final_predicted_df.at[user_id, item_id] = pred_rating
            else:
                # 유저가 train_df에 없는 경우, FunkSVD 예측 값 사용
                svd_pred = predicted_df.at[user_id, item_id] if pd.notna(predicted_df.at[user_id, item_id]) else 0
                final_predicted_df.at[user_id, item_id] = svd_pred

# DataFrame을 float로 변환하여 결측값 포함 최종 예측 평점 행렬 출력
final_predicted_df = final_predicted_df.astype(float)


# hybrid_predicted_df를 CSV 파일로 저장
file_path = "funkSVD_xgboost_predicted_df.csv"
final_predicted_df.to_csv(file_path, index=True)  # 인덱스 포함하여 저장

print(f"DataFrame이 로컬에 '{file_path}' 파일로 저장되었습니다.")