# sweep_config_ae.py
# (멀티모달 '오토인코더'용 Sweep 설정)

# 1. 튜닝할 하이퍼파라미터 정의
sweep_config = {
    'method': 'bayes',  # 베이지안 최적화 (가장 효율적)
    'metric': {
        'name': 'val_loss',     # 검증용 데이터의 '총 복원 오류'
        'goal': 'minimize'      # 복원 오류를 '최소화'하는 것을 목표로 함
    },
    'parameters': {
        # --- 1. 학습 파라미터 ---
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5, # 0.00001
            'max': 1e-3  # 0.001
        },
        'batch_size': {
            'values': [16, 32] # (AE는 메모리를 더 많이 사용하므로 16, 32 추천)
        },
        
        # --- 2. CNN (이미지) 브랜치 설정 ---
        'cnn_model': {
            'values': ['LeNet', 'AlexNet_Mini', 'VGG_Mini'] # VGGNet은 너무 커서 'Mini'로 대체
        },
        'cnn_latent_dim': {
            'values': [64, 128] # CNN이 압축할 특징 벡터 크기
        },

        # --- 3. RNN (시계열) 브랜치 설정 ---
        'rnn_model': {
            'values': ['LSTM', 'GRU']
        },
        'rnn_units': {
            'values': [32, 64] # RNN(인코더) 유닛 수
        },
        
        # --- 4. 앙상블 및 병목 ---
        'bottleneck_dim': {
            'values': [32, 64, 128] # (가장 중요) '관계'를 압축할 최종 벡터 크기
        }
    }
}