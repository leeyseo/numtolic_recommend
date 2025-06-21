import pandas as pd
import numpy as np
import json
import pickle
import schedule
import time
import requests
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecommendationTrainer:
    def __init__(self, base_url="http://localhost:8080/api"):
        self.base_url = base_url
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        self.user_features = None
        self.vectorizer = None
        self.scaler = None
        
    def fetch_data_from_api(self):
        """Spring Boot API에서 데이터 가져오기"""
        try:
            # 사용자 데이터
            users_response = requests.get(f"{self.base_url}/users")
            users_data = users_response.json()
            
            # 음식점 데이터
            restaurants_response = requests.get(f"{self.base_url}/restaurants/all")
            restaurants_data = restaurants_response.json()
            
            # 리뷰 로그 데이터
            reviews_response = requests.get(f"{self.base_url}/review-logs")
            reviews_data = reviews_response.json()
            
            logger.info(f"데이터 수집 완료 - 사용자: {len(users_data)}, 음식점: {len(restaurants_data)}, 리뷰: {len(reviews_data)}")
            
            return users_data, restaurants_data, reviews_data
            
        except Exception as e:
            logger.error(f"API 데이터 수집 실패: {e}")
            return None, None, None
    
    def preprocess_data(self, users_data, restaurants_data, reviews_data):
        """데이터 전처리"""
        try:
            # 사용자 ID 목록
            user_ids = [user['id'] for user in users_data]
            
            # 음식점 ID 목록
            restaurant_ids = [restaurant['id'] for restaurant in restaurants_data]
            
            # 사용자-음식점 평점 매트릭스 초기화
            rating_matrix = pd.DataFrame(index=user_ids, columns=restaurant_ids)
            
            # 리뷰 데이터프레임 생성
            review_df = pd.DataFrame([{
                'review_id': review['id'],
                'created_at': review['createdAt'],
                'rating': review['rating'],
                'content': review['content'],
                'restaurant_id': review['restaurantId'],
                'user_id': review['author']['id'],
                'username': review['author']['username']
            } for review in reviews_data])
            
            # 날짜 변환
            review_df['created_at'] = pd.to_datetime(review_df['created_at'])
            
            # 사용자별 통계 계산
            user_stats = self._calculate_user_stats(review_df)
            review_df = review_df.merge(user_stats, on='user_id')
            
            # 최신 리뷰만 유지 (사용자-음식점 조합당)
            review_df = review_df.sort_values('created_at', ascending=False).drop_duplicates(
                subset=['user_id', 'restaurant_id']
            )
            
            # 평점 매트릭스 채우기
            for _, row in review_df.iterrows():
                if row['user_id'] in rating_matrix.index and row['restaurant_id'] in rating_matrix.columns:
                    rating_matrix.at[row['user_id'], row['restaurant_id']] = row['rating']
            
            logger.info("데이터 전처리 완료")
            return rating_matrix, review_df, restaurants_data
            
        except Exception as e:
            logger.error(f"데이터 전처리 실패: {e}")
            return None, None, None
    
    def _calculate_user_stats(self, review_df):
        """사용자별 통계 계산"""
        user_total = review_df.groupby('user_id')['rating'].agg(['sum', 'count']).rename(
            columns={'sum': 'user_total_rating', 'count': 'user_total_count'}
        )
        
        user_restaurant_total = review_df.groupby(['user_id', 'restaurant_id'])['rating'].agg(['sum', 'count']).rename(
            columns={'sum': 'user_restaurant_rating', 'count': 'user_restaurant_count'}
        )
        
        review_df = review_df.merge(user_total, on='user_id')
        review_df = review_df.merge(user_restaurant_total, on=['user_id', 'restaurant_id'])
        
        # 다른 레스토랑 평균 평점 계산
        def calculate_other_avg(row):
            total_rating = row['user_total_rating'] - row['user_restaurant_rating']
            total_count = row['user_total_count'] - row['user_restaurant_count']
            return total_rating / total_count if total_count > 0 else 0
        
        review_df['user_other_avg'] = review_df.apply(calculate_other_avg, axis=1)
        
        # 최고 평점 계산
        user_restaurant_max = review_df.groupby(['user_id', 'restaurant_id'])['rating'].max().rename('max_rating')
        review_df = review_df.merge(user_restaurant_max, on=['user_id', 'restaurant_id'])
        
        # 최근 리뷰와의 시간 차이
        current_time = datetime.now()
        user_restaurant_latest = review_df.groupby(['user_id', 'restaurant_id'])['created_at'].max().rename('latest_review')
        review_df = review_df.merge(user_restaurant_latest, on=['user_id', 'restaurant_id'])
        review_df['days_since_review'] = (current_time - review_df['latest_review']).dt.total_seconds() / (60 * 60 * 24)
        
        return review_df[['user_id', 'user_total_rating', 'user_total_count', 'user_other_avg', 'max_rating', 'days_since_review']].drop_duplicates()
    
    def build_user_similarity_matrix(self, review_df):
        """사용자 유사도 매트릭스 구축"""
        try:
            # 사용자별 텍스트 특성
            user_text = review_df.groupby('user_id')['content'].apply(' '.join).reset_index()
            
            # 사용자별 수치 특성
            numeric_features = ['user_total_rating', 'user_total_count', 'user_other_avg', 'max_rating', 'days_since_review']
            user_numeric = review_df.groupby('user_id')[numeric_features].mean().reset_index().fillna(0)
            
            # 특성 결합
            user_features = pd.merge(user_text, user_numeric, on='user_id')
            
            # TF-IDF 벡터화
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            text_features = self.vectorizer.fit_transform(user_features['content'])
            
            # 수치 특성 스케일링
            self.scaler = StandardScaler()
            numeric_data = self.scaler.fit_transform(user_features[numeric_features])
            
            # 특성 결합
            numeric_sparse = csr_matrix(numeric_data)
            combined_features = hstack([numeric_sparse, text_features])
            
            # 코사인 유사도 계산
            cosine_sim = cosine_similarity(combined_features)
            self.user_similarity_matrix = pd.DataFrame(
                cosine_sim, 
                index=user_features['user_id'], 
                columns=user_features['user_id']
            )
            
            self.user_features = user_features
            logger.info("사용자 유사도 매트릭스 구축 완료")
            
        except Exception as e:
            logger.error(f"사용자 유사도 매트릭스 구축 실패: {e}")
    
    def build_item_similarity_matrix(self, restaurants_data):
        """아이템 유사도 매트릭스 구축"""
        try:
            # 레스토랑 데이터프레임 생성
            restaurant_df = pd.DataFrame(restaurants_data)
            
            # 수치형 특성
            numeric_features = restaurant_df[['rating', 'userRating']].fillna(0)
            scaler = StandardScaler()
            numeric_scaled = scaler.fit_transform(numeric_features)
            
            # 텍스트 특성
            restaurant_df['categories_text'] = restaurant_df['categories'].apply(
                lambda x: ' '.join([cat['name'] for cat in x])
            )
            restaurant_df['menu_text'] = restaurant_df['menu'].apply(lambda x: ' '.join(x))
            restaurant_df['text_features'] = restaurant_df['categories_text'] + ' ' + restaurant_df['menu_text']
            
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(max_features=1000)
            text_features = vectorizer.fit_transform(restaurant_df['text_features'])
            
            # 특성 결합
            combined_features = hstack([numeric_scaled, text_features])
            
            # 코사인 유사도 계산
            cosine_sim = cosine_similarity(combined_features)
            self.item_similarity_matrix = pd.DataFrame(
                cosine_sim,
                index=restaurant_df['id'],
                columns=restaurant_df['id']
            )
            
            logger.info("아이템 유사도 매트릭스 구축 완료")
            
        except Exception as e:
            logger.error(f"아이템 유사도 매트릭스 구축 실패: {e}")
    
    def save_models(self):
        """모델과 전처리 객체 저장"""
        try:
            models = {
                'user_similarity_matrix': self.user_similarity_matrix,
                'item_similarity_matrix': self.item_similarity_matrix,
                'user_features': self.user_features,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler,
                'last_updated': datetime.now()
            }
            
            with open('recommendation_models.pkl', 'wb') as f:
                pickle.dump(models, f)
            
            logger.info("모델 저장 완료")
            
        except Exception as e:
            logger.error(f"모델 저장 실패: {e}")
    
    def train(self):
        """전체 학습 파이프라인 실행"""
        logger.info("모델 학습 시작")
        
        # 데이터 수집
        users_data, restaurants_data, reviews_data = self.fetch_data_from_api()
        if users_data is None:
            logger.error("데이터 수집 실패로 학습 중단")
            return False
        
        # 데이터 전처리
        rating_matrix, review_df, restaurants_data = self.preprocess_data(users_data, restaurants_data, reviews_data)
        if rating_matrix is None:
            logger.error("데이터 전처리 실패로 학습 중단")
            return False
        
        # 유사도 매트릭스 구축
        self.build_user_similarity_matrix(review_df)
        self.build_item_similarity_matrix(restaurants_data)
        
        # 모델 저장
        self.save_models()
        
        logger.info("모델 학습 완료")
        return True

def train_job():
    """스케줄러에서 실행할 학습 작업"""
    trainer = RecommendationTrainer()
    trainer.train()

if __name__ == "__main__":
    # 즉시 한 번 학습 실행
    trainer = RecommendationTrainer()
    trainer.train()
    
    # 매일 자정에 재학습 스케줄링
    schedule.every().day.at("00:00").do(train_job)
    
    logger.info("추천 시스템 학습 스케줄러 시작")
    
    while True:
        schedule.run_pending()
        time.sleep(60)  # 1분마다 체크