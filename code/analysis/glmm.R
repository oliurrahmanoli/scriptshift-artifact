#!/usr/bin/env Rscript
# Pre-registered pooled model (research-plan.md §4.7, §10.2) — lme4::glmer.
#
#   success ~ condition * model + (1 | injection_task) + (1 | user_task)
#   + by-injection-task random slope for condition "if it converges"
#   + GEE (exchangeable, clustered on injection task) robustness check
#   + RQ2 refit with comprehension (descriptive, not mediation)
# Primary family = pooled C1–C3, Holm over 3.
#
# STATUS: verified on the synthetic fixture (D24.8, R 4.5.0 + lme4 + emmeans + geepack on the box);
# run on the real fast-track grid per D38. Python cross-checks (GEE + VB GLMM) in analysis/models.py.
#   Rscript analysis/glmm.R <out>/episodes_tidy.csv <out>/glmm_R [normalised_security] [pool|all]
# 4th arg: "pool" (default) keeps rows with primary_pool == TRUE (D38.3: self-generated cells out);
# "all" = sensitivity incl. self-generated models. Rows with status != "ok" are always dropped (D38.2).
# Input = the episodes_tidy.csv written by analysis/run_analysis.py (never the SYNTHETIC_ one
# for anything reported).
suppressPackageStartupMessages({ library(lme4); library(emmeans) })
have_gee <- requireNamespace("geepack", quietly = TRUE)
if (have_gee) suppressPackageStartupMessages(library(geepack)) else message("geepack not installed - skipping GEE robustness check")

args   <- commandArgs(trailingOnly = TRUE)
inp    <- if (length(args) >= 1) args[1] else stop("usage: glmm.R episodes_tidy.csv outdir [metric]")
outdir <- if (length(args) >= 2) args[2] else "glmm_R"
metric <- if (length(args) >= 3) args[3] else "normalised_security"
scope  <- if (length(args) >= 4) args[4] else "pool"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

# Reframe (research-plan §12, D24): LLM-authored condition names. The union with the old
# *_native names keeps this script working on both the reframed real data and the legacy
# SYNTHETIC fixture.
main_conds <- c("en", "bn_mt",
                "bn_llm", "banglish_llm", "banglish_translit", "codemix_llm",
                "bn_native", "banglish_native", "codemix_native")
d <- read.csv(inp, stringsAsFactors = FALSE)
if (any(as.logical(d$synthetic), na.rm = TRUE)) message("WARNING: input contains SYNTHETIC rows — not results")
d <- d[d$defense == "none" & d$condition %in% main_conds & !is.na(d[[metric]]), ]
if ("status" %in% names(d)) d <- d[d$status == "ok", ]
if (scope == "pool" && "primary_pool" %in% names(d)) d <- d[as.logical(d$primary_pool), ]
cat("scope:", scope, " rows:", nrow(d), " models:", paste(sort(unique(d$model)), collapse = ","), "\n")
# judge_aug_security is written as a mix of 0.0/1.0 and False/True (episode_rules model-error overlay);
# as.logical("0.0") is NA, so map both spellings explicitly (D51).
yv <- trimws(as.character(d[[metric]]))
d$y <- ifelse(yv %in% c("True", "TRUE", "true", "1", "1.0"), 1L,
              ifelse(yv %in% c("False", "FALSE", "false", "0", "0.0"), 0L, NA_integer_))
if (any(is.na(d$y))) stop("unrecognised values in metric column: ", paste(unique(yv[is.na(d$y)]), collapse = ","))
d$condition <- relevel(factor(d$condition, levels = intersect(main_conds, unique(d$condition))), ref = "en")
d$model     <- factor(d$model); contrasts(d$model) <- contr.sum(nlevels(d$model))
d$inj_uid   <- factor(d$inj_uid); d$user_uid <- factor(d$user_uid)
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

warn_int <- character(0)
fit_int <- withCallingHandlers(
  glmer(y ~ condition * model + (1 | inj_uid) + (1 | user_uid), data = d, family = binomial, control = ctrl),
  warning = function(w) { warn_int <<- c(warn_int, conditionMessage(w)); invokeRestart("muffleWarning") })
slope_note <- "ok"
fit_slope <- tryCatch(glmer(y ~ condition * model + (1 + condition | inj_uid) + (1 | user_uid),
                            data = d, family = binomial, control = ctrl),
                      error = function(e) { slope_note <<- paste("error:", conditionMessage(e)); NULL },
                      warning = function(w) { slope_note <<- paste("warning:", conditionMessage(w)); NULL })
if (!is.null(fit_slope) && isSingular(fit_slope)) slope_note <- "singular fit"
use <- if (!is.null(fit_slope) && !isSingular(fit_slope)) fit_slope else fit_int
cat("random-slope model used:", identical(use, fit_slope), "\n")
conv <- data.frame(
  scope = scope, n_rows = nrow(d), n_models = nlevels(d$model),
  intercept_model_singular = isSingular(fit_int),
  intercept_model_warnings = if (length(warn_int)) paste(unique(warn_int), collapse = " | ") else "none",
  intercept_model_conv_messages = if (length(fit_int@optinfo$conv$lme4$messages)) paste(fit_int@optinfo$conv$lme4$messages, collapse = " | ") else "none",
  random_slope_attempt = slope_note,
  model_used = if (identical(use, fit_slope)) "random_slope" else "random_intercepts")
write.csv(conv, file.path(outdir, "glmer_convergence.csv"), row.names = FALSE)
print(conv)

primary <- function(fit, label) {
  em <- emmeans(fit, ~ condition)                    # averaged over models (sum coding)
  lv <- levels(d$condition)
  mk <- function(b, a) { v <- setNames(rep(0, length(lv)), lv); v[b] <- 1; v[a] <- -1; v }
  L <- list()
  # Reframe (§12): prefer the LLM-authored names, fall back to the legacy native names.
  bn   <- if ("bn_llm" %in% lv) "bn_llm" else if ("bn_native" %in% lv) "bn_native" else NA
  bgl  <- if ("banglish_llm" %in% lv) "banglish_llm" else if ("banglish_native" %in% lv) "banglish_native" else NA
  if (!is.na(bn) && "bn_mt" %in% lv)                      L$C1 <- mk(bn, "bn_mt")
  if (!is.na(bgl) && "banglish_translit" %in% lv)         L$C2 <- mk(bgl, "banglish_translit")
  if (!is.na(bn))                                          L$C3 <- mk(bn, "en")
  ct <- summary(contrast(em, L, adjust = "holm"), infer = TRUE)  # Holm over the 3 primary contrasts
  # CI column names differ across model classes (asymp.LCL for glmer, lower.CL for geeglm).
  lo <- if (!is.null(ct$asymp.LCL)) ct$asymp.LCL else ct$lower.CL
  hi <- if (!is.null(ct$asymp.UCL)) ct$asymp.UCL else ct$upper.CL
  data.frame(model_fit = label, contrast = ct$contrast, log_or = ct$estimate, se = ct$SE,
             or = exp(ct$estimate), or_lo = exp(lo), or_hi = exp(hi),
             p_holm = ct$p.value)
}
res <- primary(use, "glmer")
if (have_gee) {
  gee <- geeglm(y ~ condition * model, id = inj_uid, data = d[order(d$inj_uid), ],
                family = binomial, corstr = "exchangeable")
  res <- rbind(res, primary(gee, "geeglm_exchangeable"))
}
write.csv(res, file.path(outdir, "T3_pooled_or_glmer.csv"), row.names = FALSE)

# RQ2: comprehension covariate and condition x comprehended interaction (descriptive)
if ("comprehended_both" %in% names(d) && any(!is.na(d$comprehended_both))) {
  dc <- d[!is.na(d$comprehended_both), ]
  dc$comprehended <- as.integer(as.logical(dc$comprehended_both))
  f1 <- glmer(y ~ condition * model + comprehended + (1 | inj_uid) + (1 | user_uid),
              data = dc, family = binomial, control = ctrl)
  f2 <- glmer(y ~ condition * model + condition:comprehended + comprehended + (1 | inj_uid) + (1 | user_uid),
              data = dc, family = binomial, control = ctrl)
  write.csv(rbind(primary(f1, "glmer_with_comprehension"), primary(f2, "glmer_condition_x_comprehended")),
            file.path(outdir, "T6_pooled_with_comprehension_glmer.csv"), row.names = FALSE)
}
sink(file.path(outdir, "glmer_summary.txt")); print(summary(use)); sink()
cat("wrote", outdir, "\n")
