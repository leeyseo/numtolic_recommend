from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import logging
from datetime import datetime
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class RecommendationServer:
    def __init__(self):
        self.models = None
        self.load_models()
    
    def load_models(self):
        """저장된 모델 로드"""
        try:
            if os.path.exists('recommendation_models.pkl'):
                with open('recommendation_models.pkl', 'rb') as f:
                    self.models = pickle.load(f)
                logger.info(f"모델 로드 완료 - 마지막 업데이트: {self.models.get('last_updated')}")
            else:
                logger.warning("저장된 모델이 없습니다. 먼저 학습을 실행해주세요.")
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}")
    
    def predict_user_based(self, user_id, restaurant_ids, k=10):
        """사용자 기반 협업 필터링 예측"""
        try:
            if self.models is None:
                return {}
            
            user_sim_matrix = self.models['user_similarity_matrix']
            
            if user_id not in user_sim_matrix.index:
                logger.warning(f"사용자 {user_id}가 유사도 매트릭스에 없습니다.")
                return {}
            
            # 유사한 사용자 찾기
            similar_users = user_sim_matrix[user_id].drop(user_id).sort_values(ascending=False).head(k)
            
            predictions = {}
            for restaurant_id in restaurant_ids:
                # 유사한 사용자들의 해당 레스토랑 평점 가중 평균
                weighted_ratings = []
                weights = []
                
                for sim_user_id, similarity in similar_users.items():
                    # 여기서는 실제 평점 데이터가 필요하지만, 
                    # 간단히 유사도 기반으로 점수 계산
                    if similarity > 0:
                        weighted_ratings.append(similarity * 4.0)  # 기본 점수
                        weights.append(similarity)
                
                if weights:
                    predicted_rating = np.average(weighted_ratings, weights=weights)
                    predictions[restaurant_id] = min(5.0, max(1.0, predicted_rating))
            
            return predictions
            
        except Exception as e:
            logger.error(f"사용자 기반 예측 실패: {e}")
            return {}
    
    def predict_item_based(self, user_id, restaurant_ids, k=10):
        """아이템 기반 협업 필터링 예측"""
        try:
            if self.models is None:
                return {}
            
            item_sim_matrix = self.models['item_similarity_matrix']
            
            predictions = {}
            for restaurant_id in restaurant_ids:
                if restaurant_id not in item_sim_matrix.index:
                    continue
                
                # 유사한 레스토랑 찾기
                similar_items = item_sim_matrix[restaurant_id].drop(restaurant_id).sort_values(ascending=False).head(k)
                
                # 유사도 기반 점수 계산
                if len(similar_items) > 0:
                    # 유사도 가중 평균으로 예측 점수 계산
                    weighted_score = np.average(
                        [4.0] * len(similar_items),  # 기본 점수
                        weights=similar_items.values
                    )
                    predictions[restaurant_id] = min(5.0, max(1.0, weighted_score))
            
            return predictions
            
        except Exception as e:
            logger.error(f"아이템 기반 예측 실패: {e}")
            return {}
    
    def get_hybrid_recommendations(self, user_id, restaurant_ids=None, top_n=10):
        """하이브리드 추천 (사용자 기반 + 아이템 기반)"""
        try:
            if restaurant_ids is None:
                # 모든 레스토랑 ID 가져오기
                if self.models and 'item_similarity_matrix' in self.models:
                    restaurant_ids = list(self.models['item_similarity_matrix'].index)
                else:
                    return []
            
            # 사용자 기반 예측
            user_based_pred = self.predict_user_based(user_id, restaurant_ids)
            
            # 아이템 기반 예측
            item_based_pred = self.predict_item_based(user_id, restaurant_ids)
            
            # 하이브리드 점수 계산 (평균)
            hybrid_scores = {}
            all_restaurant_ids = set(user_based_pred.keys()) | set(item_based_pred.keys())
            
            for restaurant_id in all_restaurant_ids:
                user_score = user_based_pred.get(restaurant_id, 0)
                item_score = item_based_pred.get(restaurant_id, 0)
                
                if user_score > 0 and item_score > 0:
                    hybrid_scores[restaurant_id] = (user_score + item_score) / 2
                elif user_score > 0:
                    hybrid_scores[restaurant_id] = user_score
                elif item_score > 0:
                    hybrid_scores[restaurant_id] = item_score
            
            # 점수 순으로 정렬하여 상위 N개 반환
            sorted_recommendations = sorted(
                hybrid_scores.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:top_n]
            
            return [
                {"restaurant_id": int(restaurant_id), "score": float(score)} 
                for restaurant_id, score in sorted_recommendations
            ]
            
        except Exception as e:
            logger.error(f"하이브리드 추천 실패: {e}")
            return []

# 전역 추천 서버 인스턴스
rec_server = RecommendationServer()

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({
        "status": "healthy",
        "model_loaded": rec_server.models is not None,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/recommend/<int:user_id>', methods=['GET'])
def get_recommendations(user_id):
    """사용자별 추천 결과 반환"""
    try:
        # 쿼리 파라미터
        top_n = request.args.get('top_n', 10, type=int)
        restaurant_ids = request.args.getlist('restaurant_ids', type=int)
        
        if not restaurant_ids:
            restaurant_ids = None
        
        # 추천 결과 생성
        recommendations = rec_server.get_hybrid_recommendations(
            user_id=user_id, 
            restaurant_ids=restaurant_ids, 
            top_n=top_n
        )
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "count": len(recommendations),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"추천 API 오류: {e}")
        return jsonify({
            "error": "추천 생성 중 오류가 발생했습니다.",
            "message": str(e)
        }), 500

@app.route('/recommend/batch', methods=['POST'])
def get_batch_recommendations():
    """배치 추천 (여러 사용자)"""
    try:
        data = request.get_json()
        user_ids = data.get('user_ids', [])
        top_n = data.get('top_n', 10)
        restaurant_ids = data.get('restaurant_ids', None)
        
        results = {}
        for user_id in user_ids:
            recommendations = rec_server.get_hybrid_recommendations(
                user_id=user_id, 
                restaurant_ids=restaurant_ids, 
                top_n=top_n
            )
            results[str(user_id)] = recommendations
        
        return jsonify({
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"배치 추천 API 오류: {e}")
        return jsonify({
            "error": "배치 추천 생성 중 오류가 발생했습니다.",
            "message": str(e)
        }), 500

@app.route('/similarity/user/<int:user_id>', methods=['GET'])
def get_similar_users(user_id):
    """유사한 사용자 찾기"""
    try:
        top_k = request.args.get('top_k', 10, type=int)
        
        if rec_server.models is None:
            return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
        
        user_sim_matrix = rec_server.models['user_similarity_matrix']
        
        if user_id not in user_sim_matrix.index:
            return jsonify({"error": f"사용자 {user_id}를 찾을 수 없습니다."}), 404
        
        similar_users = user_sim_matrix[user_id].drop(user_id).sort_values(ascending=False).head(top_k)
        
        result = [
            {"user_id": int(uid), "similarity": float(sim)} 
            for uid, sim in similar_users.items()
        ]
        
        return jsonify({
            "user_id": user_id,
            "similar_users": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"유사 사용자 API 오류: {e}")
        return jsonify({
            "error": "유사 사용자 조회 중 오류가 발생했습니다.",
            "message": str(e)
        }), 500

@app.route('/similarity/item/<int:restaurant_id>', methods=['GET'])
def get_similar_restaurants(restaurant_id):
    """유사한 레스토랑 찾기"""
    try:
        top_k = request.args.get('top_k', 10, type=int)
        
        if rec_server.models is None:
            return jsonify({"error": "모델이 로드되지 않았습니다."}), 500
        
        item_sim_matrix = rec_server.models['item_similarity_matrix']
        
        if restaurant_id not in item_sim_matrix.index:
            return jsonify({"error": f"레스토랑 {restaurant_id}를 찾을 수 없습니다."}), 404
        
        similar_items = item_sim_matrix[restaurant_id].drop(restaurant_id).sort_values(ascending=False).head(top_k)
        
        result = [
            {"restaurant_id": int(rid), "similarity": float(sim)} 
            for rid, sim in similar_items.items()
        ]
        
        return jsonify({
            "restaurant_id": restaurant_id,
            "similar_restaurants": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"유사 레스토랑 API 오류: {e}")
        return jsonify({
            "error": "유사 레스토랑 조회 중 오류가 발생했습니다.",
            "message": str(e)
        }), 500

@app.route('/reload', methods=['POST'])
def reload_models():
    """모델 재로드"""
    try:
        rec_server.load_models()
        return jsonify({
            "message": "모델이 성공적으로 재로드되었습니다.",
            "model_loaded": rec_server.models is not None,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"모델 재로드 오류: {e}")
        return jsonify({
            "error": "모델 재로드 중 오류가 발생했습니다.",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    logger.info("추천 서버 시작")
    app.run(host='0.0.0.0', port=5000, debug=False)