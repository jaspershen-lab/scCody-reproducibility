# Significance marks shared by the manuscript figures.
#
# Each scCody arm is compared with the best-performing competitor (scNODE,
# scDiffEq or ARTEMIS), chosen separately for every metric, panel, age side and
# lineage from the plotted means. Tests are two-sided paired t-tests over the
# matched replicates (fit x external age for reconstruction/DE metrics) and are
# not corrected for multiple comparisons.
#   * p < 0.05   ** p < 0.01   *** p < 0.001   **** p < 0.0001
# Stars are shown only when scCody is better than the best competitor; results
# in the other direction are left unmarked (all tests are kept in the tables).

suppressPackageStartupMessages(library(dplyr))

p_stars <- function(p) {
  case_when(
    is.na(p) ~ "",
    p < 1e-4 ~ "****",
    p < 1e-3 ~ "***",
    p < 1e-2 ~ "**",
    p < 0.05 ~ "*",
    TRUE ~ ""
  )
}

# Stars only when scCody is better; no mark when it is worse.
direction_marks <- function(p, scody_better) {
  if_else(scody_better, p_stars(p), "")
}

best_competitor_tests <- function(frame, group_cols, pair_cols = c("fold", "target_age"),
                                  higher_is_better = TRUE,
                                  arms = c("scCody-G", "scCody-CT"),
                                  competitors = c("scNODE", "scDiffEq", "ARTEMIS")) {
  frame <- frame %>% filter(is.finite(value)) %>% mutate(model = as.character(model))
  frame %>%
    group_by(across(all_of(group_cols))) %>%
    group_modify(function(group, keys) {
      means <- group %>% filter(model %in% competitors) %>% group_by(model) %>%
        summarise(mean = mean(value), .groups = "drop")
      if (!nrow(means)) return(tibble())
      best <- means$model[[if (higher_is_better) which.max(means$mean) else which.min(means$mean)]]
      bind_rows(lapply(intersect(arms, unique(group$model)), function(arm) {
        pair <- inner_join(
          group %>% filter(model == arm) %>% select(all_of(pair_cols), arm_value = value),
          group %>% filter(model == best) %>% select(all_of(pair_cols), best_value = value),
          by = pair_cols
        )
        test <- if (nrow(pair) > 2 && sd(pair$arm_value - pair$best_value) > 0)
          t.test(pair$arm_value, pair$best_value, paired = TRUE) else NULL
        tibble(
          model = arm, best_competitor = best, n_pairs = nrow(pair),
          arm_mean = mean(pair$arm_value), best_mean = mean(pair$best_value),
          difference = mean(pair$arm_value - pair$best_value),
          statistic = if (is.null(test)) NA_real_ else unname(test$statistic),
          p_value = if (is.null(test)) NA_real_ else test$p.value,
          test = "paired two-sided t-test vs best competitor; unadjusted"
        )
      }))
    }) %>%
    ungroup() %>%
    mutate(stars = direction_marks(p_value, if (higher_is_better) difference > 0 else difference < 0))
}
