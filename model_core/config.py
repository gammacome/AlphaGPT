import torch
import os

class ModelConfig:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DB_URL = f"postgresql://{os.getenv('DB_USER','postgres')}:{os.getenv('DB_PASSWORD','password')}@{os.getenv('DB_HOST','localhost')}:5432/{os.getenv('DB_NAME','crypto_quant')}"

    TRADE_SIZE_USD = 1000.0
    BASE_FEE = 0.005 # 基础费率 0.5% (Swap + Gas + Jito Tip)
    INPUT_DIM = 6
    
    ABSOLUTE_LIQUIDITY_METHOD = False #could be Relative

    PILOT_MODE = False    
    # 根据 PILOT_MODE 自动切换参数
    TRAIN_STEPS      = 50   if PILOT_MODE else 1000
    BATCH_SIZE       = 16    if PILOT_MODE else 256
    MAX_FORMULA_LEN  = 2     if PILOT_MODE else 12
    SIGNAL_THRESHHOLD = .85 if PILOT_MODE else .5
    ACTIVITY_THRESHHOLD = 5 if PILOT_MODE else 1
    MIN_LIQUIDITY = 5000.0 if PILOT_MODE else 1 # 低于此流动性视为归零/无法交易
    #RELU_THRESHHOLD = 3 if PILOT_MODE else 1.5


    