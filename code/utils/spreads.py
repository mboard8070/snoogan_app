# code/utils/spreads.py
import pandas as pd

def build_credit_spread(short_row, long_row, contracts=1):
    credit = short_row['ask'] - long_row['bid']  # worst-case fill
    width = short_row['strike'] - long_row['strike']
    max_loss = (width - credit) * 100 * contracts
    max_profit = credit * 100 * contracts
    breakeven = short_row['strike'] + credit
    
    spread = {
        "short_ticker": short_row['ticker'],
        "long_ticker": long_row['ticker'],
        "credit": round(credit, 3),
        "max_profit": round(max_profit, 2),
        "max_loss": round(max_loss, 2),
        "breakeven": breakeven,
        "pop_est": round(1 - short_row['est_delta'], 3),
        "contracts": contracts
    }
    
    print(f"Snoogans just cooked up a {contracts}-lot vertical:")
    print(f"  Sell {short_row['strike']}P @ {short_row['ask']:.2f}")
    print(f"  Buy  {long_row['strike']}P @ {long_row['bid']:.2f}")
    print(f"  Credit: ${spread['credit']:.2f} → Max profit ${spread['max_profit']}")
    print(f"  Max loss ${spread['max_loss']} — still defined risk, still chill")
    print(f"  Est. POP: {spread['pop_est']*100:.1f}% — lunch money secured")
    
    return spread