# Task

Propose exactly $n_requested new factor expressions (round $round_index).

# Grammar

$grammar

# Formation-window feedback

All statistics below come from the formation window only. IC is the mean
per-session Spearman rank correlation between the factor and the forward
return, measured after an execution lag of at least one session. ICIR is the
mean of the per-session IC divided by its standard deviation, t is a
Newey-West t-statistic of the mean IC, turnover is the per-session turnover of
the top-quintile basket, and complexity counts operators and data fields plus
0.5 per numeric literal.

Most informative so far (highest |ICIR|):
$top_block

Least informative so far:
$bottom_block

Recently rejected by the validator (fix these mistakes, do not repeat them):
$rejected_block

# Output

Return a JSON object {"proposals": [...]} with exactly $n_requested items.
Each item has:
- "expression": one expression in the grammar above, different from every
  expression listed in this message;
- "rationale": one sentence on why it could predict the cross-section of
  returns;
- "economic_mechanism": the mechanism in a few words (for example "liquidity
  provision", "investor attention" or "overreaction").

Trials used so far: $n_trials_so_far. Remaining budget: $budget_remaining.
