# Stage B Data Leakage Audit Report

**Date**: December 12, 2025  
**Auditor**: AI Assistant  
**Status**: 🚨 **CRITICAL DATA LEAKAGE DETECTED AND FIXED**

---

## Executive Summary

Stage B achieved suspiciously high results (Val accuracy 99.5%, Test accuracy 84.6%, Test AUC 86.6%) due to **critical data leakage**. The model had access to **101 columns containing current game statistics**, including the game outcome itself (`home_win`, `away_win`), scores (`home_teamScore`, `away_teamScore`), and point differentials (`home_plusMinusPoints`).

### Impact
- **Before fix**: Model could see the outcome → artificially perfect predictions
- **After fix**: Only historical features → realistic performance expected (Test AUC 0.62-0.65)

---

## TASK A: Feature Column Audit

### Findings

**Total columns in Stage B dataset**: 302

**Leakage columns detected**: **101 columns**

These columns contain statistics from the **current game** (not rolling averages from past games):

#### Home Team Current Game Stats (50+ columns)
```
home_win                    ← OUTCOME OF CURRENT GAME!
home_teamScore              ← SCORE OF CURRENT GAME!
home_opponentScore          ← OPPONENT SCORE!
home_plusMinusPoints        ← POINT DIFFERENTIAL!
home_assists
home_blocks  
home_steals
home_fieldGoalsAttempted
home_fieldGoalsMade
home_fieldGoalsPercentage
home_threePointersAttempted
home_threePointersMade
home_threePointersPercentage
home_freeThrowsAttempted
home_freeThrowsMade
home_freeThrowsPercentage
home_reboundsDefensive
home_reboundsOffensive
home_reboundsTotal
home_foulsPersonal
home_turnovers
home_numMinutes
home_q1Points
home_q2Points
home_q3Points
home_q4Points
home_benchPoints
home_biggestLead
home_biggestScoringRun
home_leadChanges
home_pointsFastBreak
home_pointsFromTurnovers
home_pointsInThePaint
home_pointsSecondChance
home_timesTied
home_timeoutsRemaining
home_seasonWins
home_seasonLosses
home_coachId
home_possessions            ← COMPUTED FROM CURRENT GAME!
home_offensive_rating       ← COMPUTED FROM CURRENT GAME!
home_defensive_rating       ← COMPUTED FROM CURRENT GAME!
home_net_rating             ← COMPUTED FROM CURRENT GAME!
home_point_diff             ← COMPUTED FROM CURRENT GAME!
```

#### Away Team Current Game Stats (50+ columns)
Same pattern as home team: `away_win`, `away_teamScore`, `away_opponentScore`, etc.

### Root Cause

The merge operation in `merge_team_stats_to_games()` correctly joined TeamStatistics to Games, but **did not exclude the current game's row statistics**. When we merged on `(gameId, teamId)`, we got:
- ✅ **Rolling features** (e.g., `home_pts_last5`) - computed from past games with `.shift(1)` ← **SAFE**
- ❌ **Current game stats** (e.g., `home_teamScore`) - from the current row ← **LEAKS OUTCOME!**

---

## TASK B: Rolling Feature Validation

### Test Methodology

For 20 random games, we recomputed `home_pt_diff_last5` manually:
1. Filter TeamStatistics for `teamId == hometeamId AND gameDateTimeEst < current_game_datetime`
2. Take last 5 games
3. Compute mean of `point_diff`
4. Compare to stored feature value

### Results

✅ **VALIDATED: No temporal leakage in rolling features**

- Tested 20 random games
- 0 violations found
- Rolling features correctly use `.shift(1)` to exclude current game
- All temporal ordering validated

**Conclusion**: The rolling feature computation logic is correct. The problem is that non-rolling columns were also included.

---

## TASK C: Merge Correctness

### Findings

❌ **MERGE ISSUE CONFIRMED**

Current game statistics found in final feature set:
- `home_win`, `away_win` ← Outcome leakage
- `home_teamScore`, `away_teamScore` ← Score leakage  
- `home_plusMinusPoints`, `away_plusMinusPoints` ← Point differential leakage

### Merge Logic Review

```python
# Merge was correct:
df_merged = df_games.merge(
    df_team_stats,
    left_on=['gameId', 'hometeamId'],
    right_on=['gameId', 'teamId'],
    how='left'
)
```

The merge itself is fine - it correctly matches game+team. The problem is **column selection after merge**:

**BEFORE (WRONG)**:
```python
# preprocessing.py only excluded 17 columns
exclude_cols = [
    'gameId', 'gameDateTimeEst', 'hometeamId', 'awayteamId',
    'homeScore', 'awayScore', 'winner', 'y',
    'home_win', 'away_win',  # Only these 2 were excluded!
    ...
]
# Result: 260 features (including 101 leakage columns)
```

**AFTER (FIXED)**:
```python
# preprocessing.py now excludes ALL current game stats
exclude_cols = [
    # Metadata (17 cols)
    'gameId', 'gameDateTimeEst', ...,
    
    # HOME team current game stats (50+ cols)
    'home_win', 'home_teamScore', 'home_opponentScore',
    'home_assists', 'home_blocks', ...,
    'home_offensive_rating', 'home_net_rating', ...,
    
    # AWAY team current game stats (50+ cols)
    'away_win', 'away_teamScore', 'away_opponentScore',
    ...
]
# Result: ~120-150 features (ONLY rolling + Stage A)
```

---

## TASK D: Feature Importance Analysis

Not performed yet - will run after fix to validate that top features make sense (rolling averages, win percentages, rest days, etc. - not scores).

---

## Recommended Safe Feature Set

### Feature Categories

1. **Stage A Historical Features** (~20 features) ✅ SAFE
   - `seasonYear`
   - `home_season_games_played`, `home_season_wins`, `home_season_win_pct`
   - `away_season_games_played`, `away_season_wins`, `away_season_win_pct`
   - `home_last5_win_pct`, `away_last5_win_pct`
   - `home_days_since_last_game`, `away_days_since_last_game`
   - Delta features: `delta_season_win_pct`, `delta_last5_win_pct`, etc.

2. **Rolling Team Statistics** (~100-120 features) ✅ SAFE
   - Window 5: `home_pts_last5`, `home_pt_diff_last5`, `home_fg_pct_last5`, ...
   - Window 10: `home_pts_last10`, `home_pt_diff_last10`, `home_ortg_last10`, ...
   - Home/Away splits: `home_net_rtg_last10_home`, `away_net_rtg_last10_away`, ...

3. **Delta Features** (~15-20 features) ✅ SAFE
   - `delta_net_rtg_last10`, `delta_pt_diff_last5`, `delta_fg_pct_last10`, ...

### Total Expected Features After Fix

**~120-150 features** (down from 260)

### Features to EXCLUDE (101 columns)

All columns matching these patterns (for both home and away):
- `*_win` (unless it's `*_win_pct`)
- `*_teamScore`, `*_opponentScore`
- `*_assists`, `*_blocks`, `*_steals` (unless `*_last5` or `*_last10`)
- `*_fieldGoals*`, `*_threePointers*`, `*_freeThrows*` (unless rolling)
- `*_rebounds*` (unless rolling)
- `*_turnovers`, `*_plusMinusPoints` (unless rolling)
- `*_q1Points`, `*_q2Points`, `*_q3Points`, `*_q4Points`
- `*_benchPoints`, `*_biggestLead`, `*_leadChanges`, etc.
- `*_possessions`, `*_offensive_rating`, `*_defensive_rating`, `*_net_rating`, `*_point_diff` (unless rolling)

**Rule of thumb**: If the column name doesn't end with `_last5`, `_last10`, `_last10_home`, or `_last10_away`, and it's not a Stage A feature, it's probably leaking!

---

## Implementation Status

### ✅ Completed

1. **`audit_stage_b_leakage.py`** - Comprehensive audit script
   - Task A: Column audit (101 leakage columns identified)
   - Task B: Rolling feature validation (0 violations)
   - Task C: Merge correctness check (confirmed issue)

2. **`preprocessing.py`** - Updated with comprehensive exclusion list
   - Added 101 columns to `exclude_cols`
   - Added validation prints for feature counts

### 🔄 Next Steps

1. **Re-run Stage B pipeline**:
   ```bash
   python run_stage_b.py
   ```

2. **Validate results**:
   - Expected Test AUC: **0.62-0.65** (down from 0.866)
   - Expected Test Accuracy: **60-62%** (down from 84.6%)
   - Expected Val Accuracy: **60-62%** (down from 99.5%)

3. **Feature importance analysis**:
   - Train logistic regression on fixed features
   - Verify top features are rolling averages (not scores)

4. **Archive comparison**:
   ```bash
   python -c "from scripts.archive_manager import compare_archives, print_comparison; c = compare_archives('archives/stage_b/run_BEFORE_FIX', 'archives/stage_b/run_AFTER_FIX'); print_comparison(c)"
   ```

---

## Lessons Learned

1. **Always audit suspicious results** - 99.5% accuracy on NBA games is impossible
2. **Merge operations need explicit column filtering** - Don't assume pandas will exclude current row
3. **Validate with recomputation** - Manually recompute features from raw data to verify
4. **Use naming conventions** - Suffix `_last5`/`_last10` made it easy to identify rolling features
5. **Separate feature engineering from merge** - Don't mix rolling computation with current game stats

---

## Expected Performance After Fix

### Before Fix (with leakage)
- Val Accuracy: 99.5%
- Test Accuracy: 84.6%
- Test AUC: 0.866
- Test Log-Loss: 2.08 (extremely bad - model overconfident)

### After Fix (no leakage) - Projected
- Val Accuracy: 60-62%
- Test Accuracy: 60-62%
- Test AUC: 0.62-0.65
- Test Log-Loss: 0.65-0.70 (reasonable)

### Comparison to Stage A
- Stage A Test AUC: 0.598
- Stage B Target: 0.62-0.65
- Expected improvement: +2-5 AUC points from team statistics

---

## Conclusion

The Stage B pipeline contained **critical data leakage** where the model had access to game outcomes and current game statistics. This has been identified and fixed by:

1. Explicitly excluding 101 columns containing current game stats
2. Keeping only Stage A features + rolling features (_last5, _last10)
3. Validating that rolling features use correct temporal ordering

The fix is implemented in `scripts/preprocessing.py`. Re-running the pipeline should now produce realistic results (~0.62-0.65 AUC).

**Status**: 🟢 **FIXED - Ready for re-execution**
