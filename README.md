# Snoogans: A Disciplined 0DTE Options Trading Bot

**Snoogans** is a reinforcement learning-based trading agent designed exclusively for vertical credit spreads on SPX and NDX indices, focusing on 0DTE and 1DTE expirations. Built with strict risk controls and a conservative growth philosophy, the bot aims for consistent, low-drawdown returns while maintaining a unique, engaging personality inspired by filmmaker Kevin Smith.

## Project Overview

Snoogans combines defined-risk options strategies with a carefully trained Proximal Policy Optimization (PPO) agent to execute high-probability, short-duration trades. The system prioritizes capital preservation, disciplined risk management, and gradual position scaling—delivering steady compounding without emotional decision-making.

For enhanced contextual awareness and natural communication, the bot leverages **Retrieval-Augmented Generation (RAG)** powered by a locally-run **Ollama instance of the Gemma 2 model**. This setup keeps everything private, fast, and offline while giving Snoogans the ability to reference historical trades, market context, and its own Jersey-flavored wisdom on the fly.

## Strategy

- **Instrument Focus**: Vertical credit spreads on SPX and NDX only
- **Expiration**: 0DTE and 1DTE
- **Primary Setup**: Short ~16-delta put credit spreads (theta-positive, high probability of profit)
- **Directional Flexibility**: Switches to call credit spreads when implied volatility and market conditions warrant
- **Risk Profile**: Strictly defined-risk positions—no naked options

## Risk Management

- 50% profit target on each trade
- 2.0–2.2× initial credit stop-loss
- Hard close at 4:00 PM EST (no overnight exposure)
- Daily max loss limit of 3% (automatic shutdown)
- Conservative position sizing: begins at 1–3 contracts, increases gradually with account growth

## Reinforcement Learning Architecture

The core trading agent is built on PPO with:
- Pre-trained checkpoints for stable initialization
- Fine-tuning focused on risk-adjusted performance
- Reward function emphasizing Sortino and Calmar ratios
- Regularization and policy constraints to prevent overfitting
- A deliberate, low-stress training regimen for robust, calm decision-making

## Personality & Communication Style

Snoogans communicates with the distinctive voice and humor of Kevin Smith—think Red Bank wit, casual Jersey phrasing, and frequent Clerks-era references (e.g., “just sold another tiny spread, lunch money secured, snoochie boochies”). This persona, brought to life through the local Gemma 2 + RAG pipeline, keeps updates entertaining and relatable while never compromising clarity on trade rationale and performance.

## Growth & Deployment Plan

- Initial phase: Paper trading at minimum size (1 contract)
- Transition to live capital only after sustained positive performance
- Gradual contract scaling as equity curve demonstrates consistency
- Target: 8–18% average monthly returns on a $40k starting account, with minimal drawdowns

## Community & Monitoring

- **Discord Server**: Primary hub for real-time trade alerts, daily/weekly recaps, performance discussion, and full-personality commentary from Snoogans.
- **Streamlit Dashboard**: Live view of equity curve, trade log, risk metrics, and scrolling bot messages—always open for quick checks.

No additional platforms or channels—everything lives cleanly between Discord and the Streamlit interface.

## Mission

To create the most disciplined, entertaining, and consistently profitable 0DTE trading system available—one that compounds capital responsibly while feeling like a friend you'd happily share a victory cigar with after another green day.

Snoochie boochies.