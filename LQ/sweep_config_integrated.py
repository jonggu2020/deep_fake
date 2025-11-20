# sweep_config_integrated.py
# (XGBoost + Tabular AE + LSTM/GRU AE 통합 튜닝 설정)

sweep_config = {
    'method': 'bayes',  # 베이지안 최적화 (똑똑하게 좋은 조합을 찾아감)
    'metric': {
        'name': 'global_score', # 낮을수록 좋은 점수 (Loss 합산)
        'goal': 'minimize'
    },
    'parameters': {
        # --- [Common] 딥러닝 학습 파라미터 ---
        'dl_learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 0.0001, 'max': 0.01
        },
        'dl_batch_size': {'values': [64, 128, 256]},
        
        # --- [Model 1] XGBoost 파라미터 ---
        'xgb_n_estimators': {'values': [100, 200, 300]},
        'xgb_max_depth': {'values': [3, 5, 7]},
        'xgb_learning_rate': {'values': [0.01, 0.05, 0.1, 0.2]},
        
        # --- [Model 2] Tabular AE (정형 데이터) 파라미터 ---
        'tab_latent_dim': {'values': [32, 64, 128]}, 
        
        # --- [Model 3] RNN AE (시계열 NPY) 파라미터 ---
        'rnn_type': {'values': ['LSTM', 'GRU']}, 
        'rnn_hidden_dim': {'values': [64, 128]},
        'rnn_layers': {'values': [1, 2]}
    }
}