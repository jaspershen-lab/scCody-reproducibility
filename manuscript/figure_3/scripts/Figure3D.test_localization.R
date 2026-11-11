#!/usr/bin/env Rscript

# Donor-paired comparison of scCody-G/CT with each dynamic comparator on the
# 71 external AIDA donors: paired two-sided t-test of absolute errors (MAE) and
# Williams' test for overlapping dependent correlations (Pearson r). The
# "best dynamics" table compares against the best comparator chosen separately
# per metric on the external results.
# Run from the repository root after assemble_localization.R:
#   Rscript manuscript/figure_3/scripts/Figure3D.test_localization.R

suppressPackageStartupMessages({library(dplyr); library(purrr); library(readr)})

result_dir <- file.path(normalizePath("."), "manuscript/figure_3/results")
scody_models <- c("scCody-G", "scCody-CT")
dynamic_models <- c("scNODE", "scDiffEq", "ARTEMIS")
source_label <- "internally selected scNODE, scDiffEq and ARTEMIS"

paired_t <- function(scody, dynamic) {
  result <- t.test(scody, dynamic, paired = TRUE, alternative = "two.sided")
  list(
    statistic = unname(result$statistic),
    p_value = result$p.value,
    mean_scody = mean(scody),
    mean_dynamic = mean(dynamic),
    mean_difference_scody_minus_dynamic = mean(scody - dynamic)
  )
}

# Williams (1959) test for two overlapping dependent correlations:
# cor(x, y) versus cor(x, z), with cor(y, z) also estimated from the same n.
williams_test <- function(x, y, z) {
  keep <- is.finite(x) & is.finite(y) & is.finite(z)
  x <- x[keep]
  y <- y[keep]
  z <- z[keep]
  n <- length(x)
  if (n <= 3 || sd(x) == 0 || sd(y) == 0 || sd(z) == 0) {
    return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                r_scody = NA_real_, r_dynamic = NA_real_, r_between_predictions = NA_real_,
                difference_scody_minus_dynamic = NA_real_))
  }

  r_scody <- cor(x, y)
  r_dynamic <- cor(x, z)
  r_between <- cor(y, z)
  determinant <- 1 - r_scody^2 - r_dynamic^2 - r_between^2 +
    2 * r_scody * r_dynamic * r_between
  average_r <- (r_scody + r_dynamic) / 2
  denominator <- 2 * (n - 1) / (n - 3) * determinant +
    average_r^2 * (1 - r_between)^3
  numerator <- (r_scody - r_dynamic) * sqrt((n - 1) * (1 + r_between))
  statistic <- numerator / sqrt(denominator)
  p_value <- 2 * pt(abs(statistic), df = n - 3, lower.tail = FALSE)
  list(statistic = statistic, p_value = p_value, n = n,
       r_scody = r_scody, r_dynamic = r_dynamic, r_between_predictions = r_between,
       difference_scody_minus_dynamic = r_scody - r_dynamic)
}

test_pair <- function(frame, arm, comparator, rule) {
  s <- frame %>% filter(model == arm) %>% select(donor_id, true_age, scody_age = predicted_age)
  d <- frame %>% filter(model == comparator) %>% select(donor_id, true_age, dynamic_age = predicted_age)
  stopifnot(!anyDuplicated(s$donor_id), !anyDuplicated(d$donor_id))
  pair <- inner_join(s, d, by = c("donor_id", "true_age"))
  stopifnot(nrow(pair) > 3)
  t <- paired_t(abs(pair$scody_age - pair$true_age), abs(pair$dynamic_age - pair$true_age))
  w <- williams_test(pair$true_age, pair$scody_age, pair$dynamic_age)
  bind_rows(
    tibble(version = "global", dynamic_source = source_label, cell_type = "Global", comparison = "MAE",
           scody_model = arm, dynamic_model = comparator, n = nrow(pair), scody_value = t$mean_scody,
           dynamic_value = t$mean_dynamic, difference_scody_minus_dynamic = t$mean_difference_scody_minus_dynamic,
           statistic = t$statistic, p_value = t$p_value, test = "paired two-sided t-test",
           selection_rule = rule, degrees_of_freedom = nrow(pair) - 1),
    tibble(version = "global", dynamic_source = source_label, cell_type = "Global", comparison = "Pearson r",
           scody_model = arm, dynamic_model = comparator, n = w$n, scody_value = w$r_scody,
           dynamic_value = w$r_dynamic, difference_scody_minus_dynamic = w$difference_scody_minus_dynamic,
           statistic = w$statistic, p_value = w$p_value,
           test = "Williams two-sided test for overlapping correlations",
           selection_rule = rule, degrees_of_freedom = w$n - 3)
  )
}

frame <- read_tsv(file.path(result_dir, "aida_selected_external_ensemble_per_donor.tsv"), show_col_types = FALSE) %>%
  filter(n_fold_predictions == 3, is.finite(true_age), is.finite(predicted_age)) %>%
  select(model, donor_id, true_age, predicted_age)

pairwise <- map_dfr(scody_models, function(arm) {
  map_dfr(dynamic_models, function(comparator) test_pair(frame, arm, comparator, "fixed comparator"))
})

summary <- frame %>% filter(model %in% dynamic_models) %>% group_by(model) %>%
  summarise(MAE = mean(abs(predicted_age - true_age)), pearson_r = cor(true_age, predicted_age), .groups = "drop")
choices <- c(MAE = (summary %>% arrange(MAE, model))$model[[1]],
             `Pearson r` = (summary %>% arrange(desc(pearson_r), model))$model[[1]])
best <- map_dfr(scody_models, function(arm) {
  map_dfr(names(choices), function(metric) {
    test_pair(frame, arm, choices[[metric]], "best dynamics selected separately per metric on external results") %>%
      filter(comparison == metric)
  })
})

write_tsv(best, file.path(result_dir, "aida_scCody_vs_best_dynamics_pvalues.tsv"), na = "")
write_tsv(pairwise, file.path(result_dir, "aida_scCody_vs_each_dynamics_pvalues.tsv"), na = "")
print(best, n = Inf, width = Inf)
