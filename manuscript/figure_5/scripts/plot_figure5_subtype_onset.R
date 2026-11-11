# Fig. 5e: donor between-patient IFN score vs days since symptom onset.
# Sourced from plot_figure5.R.

onset_col <- "between__Interferon α/β signaling"
onset <- read_csv(file.path(data_dir, "covid_ifn_symptom_onset_donors.csv"), show_col_types = FALSE) %>%
  filter(cohort %in% c("Ren", "COMBAT"), !is.na(days), days > 0) %>%
  mutate(cohort = factor(cohort, levels = c("Ren", "COMBAT")), score = .data[[onset_col]],
         severity_group = factor(if_else(severe_critical, "Severe", "Mild"),
                                 levels = c("Mild", "Severe")))
onset_stats <- read_csv(file.path(data_dir, "covid_ifn_symptom_onset.csv"), show_col_types = FALSE) %>%
  filter(step == "between", program == "IFN-α/β (Reactome)", test == "days", cohort %in% c("Ren", "COMBAT")) %>%
  mutate(cohort = factor(cohort, levels = levels(onset$cohort)),
         label = sprintf("ρ = %.2f, n = %d\nq = %s", estimate, n_donors, formatC(q, format = "e", digits = 0)))
severity_colors <- c("Mild" = "#7A9CC6", "Severe" = "#B2182B")
p <- ggplot(onset, aes(days, score)) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.3, colour = "grey55") +
  geom_smooth(method = "lm", formula = y ~ x, colour = "grey25", fill = "grey80", linewidth = 0.5) +
  geom_point(aes(colour = severity_group), size = 1.1, alpha = 0.9) +
  geom_text(data = onset_stats, aes(x = Inf, y = Inf, label = label), hjust = 1.05, vjust = 1.2,
            size = 1.9, family = "sans", lineheight = 0.95, inherit.aes = FALSE) +
  facet_wrap(~ cohort, ncol = 1, scales = "free") +
  scale_x_log10(breaks = c(1, 2, 5, 10, 20, 50, 100)) +
  scale_colour_manual(values = severity_colors, name = NULL) +
  labs(x = "Days since symptoms began",
       y = "Decoded between-patient IFN-α/β shift") +
  theme_fig() + theme(strip.background = element_blank(), strip.text = element_text(face = "bold", size = 7))
save_fig(p, "g_COVID_IFN_symptom_onset", 3.0, 3.6)
