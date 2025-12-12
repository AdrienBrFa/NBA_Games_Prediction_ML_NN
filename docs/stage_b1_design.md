# Stage B Design Document

## Overview

Stage B extends Stage A by integrating team-level performance statistics from `TeamStatistics.csv`. The goal is to improve prediction accuracy by incorporating detailed team performance metrics through rolling windows and home/away splits, while keeping the MLP model architecture identical to isolate the impact of feature engineering.

**Expected Performance**: Test AUC improvement from 0.598 (Stage A) to 0.62–0.65 (Stage B).

---

## Data Sources

### Primary Files
- **Games.csv**: Game-level data (72,307 games)
  - Key columns: `gameId`, `gameDateTimeEst`, `hometeamId`, `awayteamId`, `homeScore`, `awayScore`, `winner`
  
- **TeamStatistics.csv**: Team-level performance per game (144,614 rows = ~2 per game)
  - Key columns: `gameId`, `teamId`, `opponentTeamId`, `home`, `gameDateTimeEst`
  - Performance stats: 48 columns including shooting, rebounds, assists, turnovers, etc.

### Merge Strategy

**Merge Keys**: `(gameId, teamId)`
- Each game in Games.csv matches exactly 2 rows in TeamStatistics.csv (home team + away team)
- Merge home team stats: `Games.gameId == TeamStats.gameId AND Games.hometeamId == TeamStats.teamId`
- Merge away team stats: `Games.gameId == TeamStats.gameId AND Games.awayteamId == TeamStats.teamId`

**Validation**:
1. Assert each game has exactly 2 team-stat rows after merge
2. Report merge coverage (expected: ~100% for regular season games)
3. Drop games with missing team stats (log count and percentage)

---

## No-Leakage Policy

**CRITICAL**: For any game at datetime T, rolling features for a team must be computed ONLY from games strictly before T.

### Implementation:
1. Sort all team games by `gameDateTimeEst` ascending
2. For each game, compute rolling features using `.shift(1)` to exclude current game
3. For early-season games with insufficient history:
   - Use available games (e.g., 3 games for last-5 window)
   - Fill missing values with neutral defaults or training median

### Validation:
- Sample 10 random games
- For each, verify that the latest game in rolling window is strictly before game datetime
- Log assertion results in pipeline output

---

## Feature Engineering Specification

### A. Stage A Features (Reused)
All Stage A features are retained (~8-10 features):
- `home_season_win_pct`, `away_season_win_pct`
- `home_last5_win_pct`, `away_last5_win_pct`
- `rest_days_home`, `rest_days_away`
- `is_b2b_home`, `is_b2b_away`
- Possibly: streaks, head-to-head win pct

### B. Rolling Features (Core Signal)

For each team (home and away), compute rolling averages over **last 5 games** and **last 10 games**.

| Feature Category | Source Columns | Windows | Description |
|-----------------|----------------|---------|-------------|
| **Scoring** | `teamScore`, `opponentScore` | 5, 10 | Points for/against, point differential |
| **Shooting Efficiency** | `fieldGoalsPercentage`, `threePointersPercentage`, `freeThrowsPercentage` | 5, 10 | FG%, 3P%, FT% |
| **Rebounding** | `reboundsTotal`, `reboundsOffensive`, `reboundsDefensive` | 5, 10 | Total, offensive, defensive rebounds |
| **Ball Movement** | `assists`, `turnovers` | 5, 10 | Assists, turnovers, assist-to-turnover ratio |
| **Defense** | `blocks`, `steals`, `opponentScore` | 5, 10 | Defensive stats |
| **Pace** | `possessions` (derived) | 5, 10 | Estimated possessions = FGA - ORB + TO + 0.4*FTA |
| **Advanced** | Offensive/Defensive Rating (derived) | 5, 10 | ORtg = 100 * PTS / Poss, DRtg = 100 * Opp_PTS / Poss |

**Total Rolling Features**: ~16-20 per team x 2 windows = 32-40 features

#### Key Rolling Features to Implement:

**Home Team (prefix `home_`):**
- `home_pts_last5`, `home_pts_last10`
- `home_opp_pts_last5`, `home_opp_pts_last10`
- `home_pt_diff_last5`, `home_pt_diff_last10`
- `home_fg_pct_last5`, `home_fg_pct_last10`
- `home_3p_pct_last5`, `home_3p_pct_last10`
- `home_ft_pct_last5`, `home_ft_pct_last10`
- `home_reb_total_last5`, `home_reb_total_last10`
- `home_ast_last5`, `home_ast_last10`
- `home_tov_last5`, `home_tov_last10`
- `home_stl_last5`, `home_stl_last10`
- `home_blk_last5`, `home_blk_last10`
- `home_ortg_last5`, `home_ortg_last10`
- `home_drtg_last5`, `home_drtg_last10`
- `home_net_rtg_last5`, `home_net_rtg_last10` (ORtg - DRtg)
- `home_pace_last5`, `home_pace_last10`

**Away Team (prefix `away_`):**
- Same 15 features as home team with `away_` prefix

### C. Home/Away Split Features

Compute rolling features separately for home and away games to capture location effects.

| Feature | Window | Filter | Description |
|---------|--------|--------|-------------|
| `home_net_rtg_L10_home` | 10 | home==1 | Home team's net rating in last 10 home games |
| `home_net_rtg_L10_away` | 10 | home==0 | Home team's net rating in last 10 away games |
| `away_net_rtg_L10_home` | 10 | home==1 | Away team's net rating in last 10 home games |
| `away_net_rtg_L10_away` | 10 | home==0 | Away team's net rating in last 10 away games |
| `home_fg_pct_L10_home` | 10 | home==1 | Home team's FG% in last 10 home games |
| `away_fg_pct_L10_away` | 10 | home==0 | Away team's FG% in last 10 away games |
| `home_pt_diff_L10_home` | 10 | home==1 | Home team's point diff in last 10 home games |
| `away_pt_diff_L10_away` | 10 | home==0 | Away team's point diff in last 10 away games |

**Total Split Features**: ~8 features (4 per team)

### D. Delta Features (Matchup Contrasts)

Compute home minus away for key metrics to give the model direct comparison signals.

| Feature | Calculation | Description |
|---------|-------------|-------------|
| `delta_net_rtg_last10` | `home_net_rtg_last10 - away_net_rtg_last10` | Net rating differential |
| `delta_pt_diff_last5` | `home_pt_diff_last5 - away_pt_diff_last5` | Recent point diff comparison |
| `delta_fg_pct_last10` | `home_fg_pct_last10 - away_fg_pct_last10` | Shooting efficiency gap |
| `delta_3p_pct_last10` | `home_3p_pct_last10 - away_3p_pct_last10` | Three-point shooting gap |
| `delta_tov_last5` | `home_tov_last5 - away_tov_last5` | Turnover differential |
| `delta_reb_last10` | `home_reb_total_last10 - away_reb_total_last10` | Rebounding advantage |
| `delta_ast_last10` | `home_ast_last10 - away_ast_last10` | Assist differential |
| `delta_pace_last10` | `home_pace_last10 - away_pace_last10` | Pace differential |

**Total Delta Features**: ~8 features

---

## Total Feature Count

| Category | Count |
|----------|-------|
| Stage A features | ~8-10 |
| Rolling features (home + away) | ~30-40 |
| Home/away split features | ~8 |
| Delta features | ~8 |
| **TOTAL** | **~54-66 features** |

Target range: **55-65 features** (intermediate complexity)

---

## Missing Value Policy

### Early Season Games
For games where a team has fewer than N games in history (e.g., first 5 games of season):
1. Compute rolling on available games only
2. If 0 games available (season opener): use neutral defaults
   - Point differential: 0
   - Percentages (FG%, 3P%, FT%): league average or 0.45/0.35/0.75
   - Counts (rebounds, assists): league average

### Implementation
Use `SimpleImputer(strategy='median')` on training set, apply to val/test.

---

## Model Configuration

**Model**: Same MLP as Stage A
- Architecture: Input(~55-65) → Dense(64, relu, L2=0.001) → Dropout(0.3) → Dense(32, relu, L2=0.001) → Dropout(0.3) → Dense(1, sigmoid)
- Loss: Binary crossentropy
- Optimizer: Adam
- Metrics: Accuracy, AUC, F1
- Callbacks: EarlyStopping(patience=5), ReduceLROnPlateau(patience=3), ModelCheckpoint

**Why unchanged?**: To isolate feature improvements. If Stage B improves performance, we know it's due to better features, not model changes.

---

## Expected Outcomes

### Performance Targets
- **Stage A baseline**: Test AUC = 0.598, Accuracy = 58.55%
- **Stage B target**: Test AUC = 0.62–0.65, Accuracy = 60–62%

### Hypothesis
Rolling team statistics capture:
1. **Recent form**: Last 5/10 games show current team strength
2. **Matchup quality**: Delta features directly compare team strengths
3. **Home/away effects**: Split features capture location-specific performance
4. **Momentum**: Trends in shooting, turnovers, pace predict future performance

### Failure Modes to Monitor
- **Overfitting**: Val AUC >> Test AUC (mitigate with L2/Dropout)
- **Feature leakage**: Accidentally using current game stats (prevent with strict datetime filtering)
- **Multicollinearity**: Too many correlated rolling features (use correlation analysis)

---

## Implementation Checklist

- [x] Inspect TeamStatistics.csv schema
- [x] Define merge keys and validation logic
- [x] Document feature specifications
- [ ] Implement `features/stage_b_teamstats.py`
- [ ] Integrate into `run_stage_b.py`
- [ ] Add leakage validation tests
- [ ] Update README.md
- [ ] Execute and archive first Stage B run
- [ ] Compare Stage A vs B1 performance

---

## Notes

- TeamStatistics.csv contains **48 columns** of rich team performance data
- Each game has exactly 2 rows (home + away team)
- `gameDateTimeEst` is consistent between Games.csv and TeamStatistics.csv
- Data spans from historical games through current season (2025-12-08 in latest row)
- Regular season filtering remains critical (no playoffs in training)
