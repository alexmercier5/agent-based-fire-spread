# agent-based-fire-spread
GT 8900 Research Project developing an agent based fire spread model to validate and test against FlamMap. 

TODO: 
--Improve computational efficiency. Approximately 45 seconds to run full sim. Can reduce the comp. time by not iterating through each cell agent for every step. Can also remove burned cells from evaluation of neighbors.
--Need to figure out why right side is straight. Could be due to neighbor search (common pattern) or something else.
--Review default values -- ensure fire ignition point is accurate to fire .shp file from FlamMap
    --In fire agent ensure that the Rothermel eqn uses as many neighbor values as possible rather than the default fire object values
--Compile arrival times for all cell agents in data frame to compare with arrival times from FlamMap
--Determine where inaccuracies are and root causes. Incorporate fixes or be able to explain why.

Optional:
--Add ability to evaluate multiple fire ignition points at the same time. May already be working but can double check later.
--Ensure wind factor and slope factors are working properly.
--Add additionaly TIF layers 6 through 8 to Rothermel equation. Must be added to cell agents for this to work. 
