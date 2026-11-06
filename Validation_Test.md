# Midterm Validation Test Results

Final settings were used unchanged for validation. Measurements were taken after the robot stopped, using the same marked chassis reference point and starting pose each time.

## Session conditions

- Date: 11/5/26
- Floor / test location: Lab floor in Observatory Mountain Engineering Research Facility
- Battery voltage at start: Full
- Distance measurement tool / smallest marked increment: Tape Measure (0.1 cm increment)
- Angle measurement method / smallest marked increment: Pocket-size protractor (single unit of degree)
- Start: middle of the front wheel, lined up with the tape.
- End: middle of the robot, measured to the tape.

Sign convention: + for left drift and left heading error, − for right. For the left-turn test, actual left rotation is positive. Statistics use the sample standard deviation (n − 1).

------------------------------------------------------------------------

## Summary

| Test | Target | n | Mean measured | Mean signed error | Mean abs. error | Sample SD |
|----|----|----|----|----|----|----|
| 1\. Short straight (0.05 m/s, 6.9 s) | 3 m | 5 | 3.066 m | +0.066 m (+2.2 %) | 0.066 m | 0.021 m |
| 2\. Left turn (+0.2 rad/s, 1.92 s) | 90° | 5 | 86.1° | −3.9° (−4.3 %) | 3.9° | 1.14° |
| 3\. Long straight (0.05 m/s, 15.4 s) | 7 m | 5 | 6.994 m | −0.006 m (−0.09 %) | 0.022 m | 0.030 m |
| 4\. Combined AUTO route | finish point | 5 | mean endpoint error 0.469 m | n/a | n/a | 0.059 m |

------------------------------------------------------------------------

## 1. Short straight, five runs

**Request:** linear 0.05, angular 0; **duration:** 6.9 s; **target:** 3 m.

| Run | Forward displacement (m) | Distance error (m) | Distance error (%) |   |   |   |   |
|----|----|----|----|----|----|----|----|
| 1 | 3.04 | +0.04 | +1.33 |  |  |  |  |
| 2 | 3.05 | +0.05 | +1.67 |  |  |  |  |
| 3 | 3.07 | +0.07 | +2.33 |  |  |  |  |
| 4 | 3.09 | +0.09 | +3.00 |  |  |  |  |
| 5 | 3.08 | +0.08 | +2.67 |  |  |  |  |

**Statistics**

| Metric                    | Value             |
|---------------------------|-------------------|
| Mean forward displacement | 3.066 m           |
| Mean signed error         | +0.066 m (+2.2 %) |
| Mean absolute error       | 0.066 m           |
| Sample standard deviation | 0.021 m           |
| Range                     | 3.04 – 3.09 m     |

All five runs overshot the target.

------------------------------------------------------------------------

## 2. Left turn, five runs

**Request:** linear 0, angular +0.2; **duration:** 1.92 s; **target:** left 90°.

| Run | Actual left rotation (°) | Turn error (°) | Turn error (%) |
|-----|--------------------------|----------------|----------------|
| 1   | 85.0                     | −5.0           | −5.56          |
| 2   | 87.5                     | −2.5           | −2.78          |
| 3   | 85.0                     | −5.0           | −5.56          |
| 4   | 87.0                     | −3.0           | −3.33          |
| 5   | 86.0                     | −4.0           | −4.44          |

**Statistics**

| Metric                    | Value          |
|---------------------------|----------------|
| Mean left rotation        | 86.1°          |
| Mean signed error         | −3.9° (−4.3 %) |
| Mean absolute error       | 3.9°           |
| Sample standard deviation | 1.14°          |
| Range                     | 85.0 – 87.5°   |

All five runs undershot the target.

------------------------------------------------------------------------

## 3. Long straight, five runs

**Request:** linear 0.05, angular 0; **duration:** 15.4 s; **target:** 7 m.

| Run | Forward displacement (m) | Distance error (m) | Distance error (%) | Sideways drift (m, signed) | Drift / distance (%) |   |   |   |
|----|----|----|----|----|----|----|----|----|
| 1 | 6.95 | −0.05 | −0.71 | +0.165 | 2.37 |  |  |  |
| 2 | 7.02 | +0.02 | +0.29 | +0.200 | 2.85 |  |  |  |
| 3 | 7.00 | 0.00 | 0.00 | +0.165 | 2.36 |  |  |  |
| 4 | 6.98 | −0.02 | −0.29 | +0.090 | 1.29 |  |  |  |
| 5 | 7.02 | +0.02 | +0.29 | +0.190 | 2.71 |  |  |  |

**Statistics**

| Metric | Forward displacement | Sideways drift |
|----|----|----|
| Mean | 6.994 m | +0.162 m (about 2.3 % of distance) |
| Mean signed error | −0.006 m (−0.09 %) | n/a |
| Mean absolute error | 0.022 m | n/a |
| Sample standard deviation | 0.030 m | 0.043 m |
| Range | 6.95 – 7.02 m | +0.090 – +0.200 m |

Drift was to the left in every run.

------------------------------------------------------------------------

## 4. Combined AUTO route, five runs

**Sequence:** 3 m forward → left 90° → 7 m forward, with the existing 0.5 s stops.

The sheet specifies three runs. Five were recorded, and all five are included. Heading errors were entered without a sign and are treated here as magnitudes.

| Run | Finish error, forward axis (m) | Finish error, left axis (m) | Endpoint error (m) | Final heading error (°) |   |
|----|----|----|----|----|----|
| 1 | +0.220 | +0.290 | 0.364 | 2 |  |
| 2 | +0.265 | +0.410 | 0.488 | 3 |  |
| 3 | +0.270 | +0.430 | 0.508 | 3 |  |
| 4 | +0.220 | +0.440 | 0.492 | 5 |  |
| 5 | +0.270 | +0.410 | 0.491 | 4 |  |

Endpoint error = sqrt(forward-axis error² + left-axis error²).

**Statistics**

| Metric              | Mean     | Sample SD | Range             |
|---------------------|----------|-----------|-------------------|
| Endpoint error      | 0.469 m  | 0.059 m   | 0.364 – 0.508 m   |
| Forward-axis error  | +0.249 m | 0.027 m   | +0.220 – +0.270 m |
| Left-axis error     | +0.396 m | 0.061 m   | +0.290 – +0.440 m |
| Final heading error | 3.4°     | 1.14°     | 2 – 5°            |

Using only runs 1–3 (the sheet's required count), the mean endpoint error is 0.453 m.

------------------------------------------------------------------------

## Observations

- **Straight-line distance is well tuned.** The long straight averages within 0.1 % of target. The short straight overshoots by about 2 %. Both are repeatable to within about 3 cm.
- **The turn is the weakest calibration.** Every run undershot, with a mean error of −3.9°. Extending the turn duration by about 4.3 % (1.92 s to roughly 2.0 s) is a rough estimate of what would center it near 90°. That figure is extrapolated from this data and is not verified. If the setting is changed, label the new runs as calibration and start a new validation set.
- **The robot drifts left on straights.** Mean drift is about 16 cm over 7 m.
- **Combined-route error comes mainly from the turn undershoot and the left drift.** Both place the robot ahead of and to the left of the intended finish. The positive forward and left errors are consistent in every run.
