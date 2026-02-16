import torch
from .config import ModelConfig

class MemeBacktest:
    def __init__(self):
        self.trade_size = 1000.0
        self.min_liq = 500000.0
        self.base_fee = 0.0060
        self.base_fee = 0.0020

    def evaluate(self, factors, raw_data, target_ret):
        #print(__file__)  
        liquidity = raw_data['liquidity']
        signal = torch.sigmoid(factors)
          
        median_liq = liquidity.median()
        is_safe = (liquidity > self.min_liq).float() if ModelConfig.ABSOLUTE_LIQUIDITY_METHOD else (liquidity > median_liq * 0.1).float()
        
        position = (signal > ModelConfig.SIGNAL_THRESHHOLD).float() * is_safe

        #debug
        is_safe = torch.ones(signal.shape, device=signal.device)
    
        position = (signal > 0.5).float() * is_safe
        #print(f"signal>0.5 sum: {(signal > 0.5).float().sum()}")
        #print(f"is_safe sum: {is_safe.sum()}")
        #print(f"position sum: {position.sum()}")
        #debug end
        impact_slippage = self.trade_size / (liquidity + 1e-9)
        impact_slippage = torch.clamp(impact_slippage, 0.0, 0.05)
        total_slippage_one_way = self.base_fee + impact_slippage
        prev_pos = torch.roll(position, 1, dims=1)
        prev_pos[:, 0] = 0
        turnover = torch.abs(position - prev_pos)
        tx_cost = turnover * total_slippage_one_way
        gross_pnl = position * target_ret
        net_pnl = gross_pnl - tx_cost
        cum_ret = net_pnl.sum(dim=1)
        big_drawdowns = (net_pnl < -0.05).float().sum(dim=1)
        # 换手率惩罚
        turnover_penalty = turnover.mean(dim=1) * 50.0
        #score = cum_ret - (big_drawdowns * 2.0)
        score = cum_ret - (big_drawdowns * 2.0) - turnover_penalty
        #added end  
        activity = position.sum(dim=1)  # 每个币的持仓次数

        #added by yf
        # 太少开仓 → 惩罚
        score = torch.where(activity < 10, torch.tensor(-10.0, device=score.device), score)

        # 太多开仓（超过70%时间都持仓）→ 也惩罚  
        score = torch.where(activity > 700, torch.tensor(-10.0, device=score.device), score)
        #added end

        #debug
        #print(f"activity: {activity} | signal range: {signal.min():.3f}~{signal.max():.3f}")
        #print(f"signal shape: {signal.shape}")
        #print(f"liquidity shape: {liquidity.shape}")
        #print(f"is_safe shape: {is_safe.shape}")
        #print(f"activity: {activity.mean():.1f}")
        #debug end
        #score = torch.where(activity < ModelConfig.ACTIVITY_THRESHHOLD, torch.tensor(-10.0, device=score.device), score)
        

        final_fitness = torch.median(score)

        print(f"activity: {activity} | cum_ret: {cum_ret} | turnover_penalty: {turnover_penalty} | score: {score}")
        return final_fitness, cum_ret.mean().item()