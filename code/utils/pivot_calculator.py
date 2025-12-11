# code/utils/pivot_calculator.py — FREE TIER FINAL (real pivots, no Polygon aggs)
import yfinance as yf
from datetime import datetime

def get_pivots():
    message = "**SNOOGANS' DAILY PIVOTS — 8:30 AM EST**\n\n"
    
    for symbol, name in [("^GSPC", "SPX"), ("^NDX", "NDX")]:
        try:
            data = yf.download(symbol, period="5d", progress=False)
            if len(data) < 2:
                raise Exception("Not enough data")
            yesterday = data.iloc[-2]  # second-to-last row = yesterday
            h, l, c = yesterday['High'], yesterday['Low'], yesterday['Close']
            
            pp = (h + l + c) / 3
            r1 = 2*pp - l
            s1 = 2*pp - h
            r2 = pp + (h - l)
            s2 = pp - (h - l)
            r3 = h + 2*(pp - l)
            s3 = l - 2*(h - pp)
            
            message += f"**{name}**  H:${h:.0f} L:${l:.0f} C:${c:.0f}\n"
            message += f"R3: ${r3:.0f}\n"
            message += f"R2: ${r2:.0f}\n"
            message += f"R1: ${r1:.0f}\n"
            message += f"PP: ${pp:.0f}\n"
            message += f"S1: ${s1:.0f}\n"
            message += f"S2: ${s2:.0f}\n"
            message += f"S3: ${s3:.0f}\n\n"
        except:
            message += f"**{name}**: Data unavailable — market's sleeping in\n\n"
    
    message += "Trade above R1, fade below S1 — lunch money secured. 37."
    return message

# Test
if __name__ == "__main__":
    print(get_pivots())