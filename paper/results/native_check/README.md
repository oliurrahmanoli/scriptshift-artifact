# Author native-speaker check (2026-10-14)

* `native_check_ratings_returned_raw.csv`: the sheet as returned by the author. Its Bengali-script payload text was
  replaced by `?` characters when the file was saved (non-UTF-8 export); ids, English sources and ratings are intact.
* `native_check_ratings.csv`: the same ratings joined by `id` to the original UTF-8 payload text
  (from `code/data/payloads.json`).
* Ratings: Natural 1-3 (3 = natural), Meaning preserved correct/drifted. Not blind, not independent.
* `native_check_ratings_printout.pdf`: the author's PDF printout of his sheet (received 2026-10-14). It shows 11 ratings
  that were blank in the CSV/xlsx exports (bn_llm naturalness for banking/injection_task_2-8 and slack/injection_task_1-3,
  page 3; bn_mt:banking/injection_task_5 meaning = correct, page 9). All other ratings match the CSV.
* `native_check_ratings_merged.csv`: the authoritative, complete ratings (CSV ratings + the 11 values read from the printout).
* 2026-10-14 second pass: the author re-checked the 11 banglish_llm items rated drifted (sheet /workspace/out/banglish_drift_recheck.xlsx) and confirmed all 11 as drifted. No rating changed.
