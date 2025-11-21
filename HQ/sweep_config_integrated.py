# sweep_config_integrated.py
# (XGBoost + Tabular AE + LSTM/GRU AE 통합 튜닝)

sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'global_score', # 모든 모델의 성능을 합산한 지표 (낮을수록 좋음)
        'goal': 'minimize'
    },
    'parameters': {
        # --- [Common] Deep Learning 학습 파라미터 ---
        'dl_learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4, 'max': 1e-2
        },
        'dl_batch_size': {'values': [64, 128]},
        
        # --- [Model 1] XGBoost 파라미터 ---
        'xgb_n_estimators': {'values': [100, 200]},
        'xgb_max_depth': {'values': [3, 5, 7]},
        'xgb_learning_rate': {'values': [0.01, 0.1, 0.2]},
        
        # --- [Model 2] Tabular AE (CSV) 파라미터 ---
        'tab_latent_dim': {'values': [32, 64]}, # 압축 차원
        
        # --- [Model 3] RNN AE (NPY) 파라미터 ---
        'rnn_type': {'values': ['LSTM', 'GRU']}, # 모델 선택
        'rnn_hidden_dim': {'values': [64, 128]},
        'rnn_layers': {'values': [1, 2]}
    }
}