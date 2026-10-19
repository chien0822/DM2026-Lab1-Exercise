# Homework Report
 
## 1. Data and setup
 
The dataset, `newdataset/Reddit-stock-sentiment.csv`, has 847 Reddit posts about stocks and 16 columns (`list_files_1`), including Reddit metadata, automatic `subjectivity`/`polarity` scores and two label columns, `sentiment` and `label`. `load_dataset_tool` keeps only the text and one label column.
 
My first load (`load_dataset_2`) returned −1.0: 197, 0.0: 331, 1.0: 319, which did not match the dataset description (315 / 423 / 109). My prompt had not named the label column, and with two candidates the tool took `sentiment`. I reloaded with `label_column="label"` (`load_dataset_4`: 315 / 423 / 109, codes −1.0 → 0, 0.0 → 1, 1.0 → 2). From then on, before using a new tool I asked the agent to list its parameters and defaults first. The two label columns disagree strongly on the positive class (319 vs 109); `sentiment` may come from a lexicon scorer, which would suggest that word lists over-detect positive sentiment here, but this was not verified.
 
The classes are imbalanced (neutral about half, positive about one eighth), which later affects `max_df`, support thresholds and how far one-vs-rest results for the positive class can be trusted.
 
**Tool access.** I approved all tools at the start, since the homework uses every one of them, they are my own implementations (plus the two provided tools), and they only work on the session's in-memory data and write plots and session artifacts. The cost is that any parameter I leave out falls back to a default without notice, which happened twice (`load_dataset_2`, `spearman_filter_23`). I handled that through my prompts, by asking for parameters and defaults first and setting every relevant one explicitly.
 
## 2. Data quality
 
`inspect_5` showed that the posts are mostly single sentences in trading slang, and some labels looked questionable to me: "Seeing lots of red in the ticker." is neutral although "red" means falling prices, "Calls on retards" is negative although calls are a bullish bet, and "He didn’t say thank you." is negative but not about stocks. This made me expect noisy labels and weak word-level signals.
 
`missing_6` found 0 missing texts. My code only checks NaN and blank strings, so placeholders such as `[removed]` would pass; neither appears in the top 20 terms (`term_freq_12`), so they occur fewer than 111 times if at all.
 
`duplicates_7` reported 23 duplicates. To see whether they were real reposts, I inspected the first 45 rows (`inspect_8`): row 42 repeats row 34 word for word with the same label, so I dropped duplicates. `duplicates_9` then removed 34 rows, not 23, because my tool drops with `keep=False` and removes every copy (about 11 texts lost entirely, 1.3% of the data); I accepted this and note it in §8. The removals were uneven: 32 of 34 were neutral (`describe_10`: 315 / 391 / 107, total 813), which suggests reposted boilerplate, although I did not verify it.
 
`describe_10` (plot: `chienhung_homework_plot_describe_10.png`) shows that the texts are extremely short: median 78 characters (mean 147, max 2,737), more than ten times shorter than the newsgroup posts. Negative posts are the longest (median 96 vs 68 neutral and 77 positive), which later inflates negative word shares (§5).
 
## 3. Document-term matrix
 
| Result | ngram | min_df | max_df | Vocabulary | Non-zero | Total count |
|---|---|---|---|---|---|---|
| `dtm_11` | 1 | 1 | 1.0 | 4,313 | 16,659 | 20,315 |
| `dtm_13` | 1 | 2 | 0.5 | 1,659 | 14,005 | 17,410 |
| `dtm_15` | 1–2 | 2 | 0.5 | 3,346 | 19,567 | 23,263 |
| `dtm_17` | 1 | 2 | 0.1 | 1,634 | 10,387 | 11,681 |
 
The baseline (`dtm_11`, `term_freq_12`) has only function words in its top 20, and total counts are only 1.22 times the non-zero cells: most words occur once per post, so the matrix is nearly binary. I expected this to weaken the variance ranking and the Pearson/Spearman difference that mattered in the newsgroups.
 
I then tried the newsgroup settings (`dtm_13`) and predicted that `max_df=0.5` would remove little. It removed nothing: to occur in over 406 posts a word needs at least 407 occurrences, and only `the`, `to` and `and` have that many, all still present. The whole reduction came from `min_df=2` (61.5% of terms occur once). Adding bigrams (`dtm_15`) left the top 20 unchanged, since no phrase repeats often in 813 short posts, so I did not keep it.
 
To push function words out I lowered `max_df` to 0.1 (`dtm_17`, `term_freq_18`). This removed only 25 terms, but a third of all occurrences, and content words appeared: `trump` 92, `china` 72, `market` 72, showing that the "stock" posts are largely about politics and the macro market. I kept `dtm_17` as the working matrix. Because the tool lists only the top 20, six removed terms were unknown, and I suspected one was the negation `not`; §7 confirms it.
 
## 4. Feature filtering
 
Before running the variance filter, I worked out that a 0/1 term with document share p ≤ 0.0996 has variance at most p(1 − p) ≈ 0.09. Threshold 0.1 (`variance_filter_19`, plot: `chienhung_homework_plot_turn114.png`) kept only 19 terms, so every kept term must be repeated inside some posts; e.g. `your` is kept while the more frequent `no` is not. The agent described them as "the highest-frequency terms"; when I pointed to `your`/`no`, it agreed. Threshold 0.01 (`variance_filter_20`) kept 410 terms, i.e. it removes words in fewer than about 9 posts.
 
| Class | Top terms (r) |
|---|---|
| Negative (`pearson_filter_21`) | his 0.167, trump 0.151, fuck 0.134, china 0.126, him 0.122, economy 0.119, trade 0.115 |
| Positive (`pearson_filter_22`) | huge / main / tldr 0.1563, nice 0.1555, technology 0.1525, potential 0.132, 3d / amd / catch / crush 0.1276 |
 
All correlations are weak (newsgroups: 0.51). Negative posts are marked by politics and profanity, not by sentiment words. The positive list looked strange: many terms share exactly the same r. For a 0/1 term, r depends only on how many posts contain it and how many of those are positive, so I solved backwards: 0.1276 is a term in 2 posts, both positive; 0.1563 is 3 posts, all positive. The positive "keywords" are therefore words seen in 2–5 posts, and their p-values are far too small (directly, a 2-post term lands in the positive class twice with probability about 1.7%, not 0.03%). With 1,634 tests, only `his` and `trump` pass a Bonferroni threshold. The agent read the positive list as "tech hype" and claimed the words came from the same posts; neither is supported by the counts.
 
I predicted that Spearman would match Pearson on near-binary data. My first run (`spearman_filter_23`) silently used the default `max_terms=1500`; I reran with all 1,634 terms (`spearman_filter_24`) and got the identical top 15, as expected, since the excluded terms can reach at most ρ ≈ 0.062. For both classes (`spearman_filter_24`, `_25`), 13–14 of 15 terms are the same, and terms that never repeat within a post get exactly the same value. Only repeated terms move: `china` falls from 4th to 15th for negative, while `technology` rises to 1st for positive (I had predicted it would fall). On this data Spearman adds almost nothing beyond Pearson.
 
## 5. Frequent pattern mining
 
`mine_patterns_tool` builds its own per-class matrix with an English stop-word list. Since `min_sup` is an absolute count, I set it to about 3% of each class: 10 for negative (`mine_patterns_26`), 12 for neutral (`_29`) and 5 for positive (`_30`, since 3 posts is too easily reached by chance). Every listed pattern was a single word. To check whether any combinations exist, I lowered the negative threshold to 5 (`mine_patterns_27`): the strongest pairs (`china`+`trump`, 9 posts) cover under 3% of the class, and maximal itemsets then hide the most common single words, so that output is misleading as a ranking.
 
| Term | Negative (315) | Neutral (391) | Positive (107) |
|---|---|---|---|
| trump | 14.3% | 5.9% | < 5 posts |
| china | 10.5% | 5.6% | < 5 posts |
| economy | 5.4% | ≤ 3.3% | 5.6% |
| tariffs | 5.1% | ≤ 3.3% | 4.7% |
| good | 4.1% | 3.8% | **11.2%** |
 
Comparing shares across classes changed my reading of §4. Nearly every word has a higher share in negative posts, which I attribute to their greater length (about 1.4× neutral); `trump` and `china` exceed that clearly. `economy` and `tariffs` are as common in positive as in negative posts, so they separate opinionated from neutral posts; they ranked high for negative in §4 only because the "rest" was mostly neutral.
 
`good` is the clearest sentiment word in the whole analysis, but it was missing from the correlation lists. From its supports I estimated r ≈ 0.11, below the 0.1276 held by two-post words: correlation rewards purity, support rewards frequency, and the two methods complement each other. The stop-word list removes `not` and `no`, leaving only fragments like `don`, so some of the 13 negative `good` posts may be "not good". Neutral held 32 of 34 duplicates, so I also checked for surviving templates: no multi-word pattern appears, so none occurs in 14 or more neutral posts.
 
## 6. Dimensionality reduction
 
I ran PCA, t-SNE and UMAP on `dtm_17` with `random_state=42` (default is `None`). PCA (`reduce_dimensions_31`, plot: `chienhung_homework_plot_reduce_dimensions_31.png`) explains 6.62% + 3.30%: a dense blob, a ray along PC1 and two extreme outliers, with all classes overlapping. Negative coordinates on PC1 show the data are centred. The most extreme points are neutral, positive, negative in that order, matching the longest post of each class (2,737, 2,444, 2,028 characters), so PC1 most likely reflects length (not verifiable with the tool).
 
t-SNE at perplexity 30 (`reduce_dimensions_32`, plot: `chienhung_homework_plot_reduce_dimensions_32.png`) gave a featureless uniform disc. To check whether this was a parameter effect, I reran with perplexity 5 (`reduce_dimensions_33`, plot: `chienhung_homework_plot_reduce_dimensions_33.png`): small mixed groups in the centre and an evenly spaced field of isolated points. I explain this from the DTM: a post has about 12.8 terms, two posts usually share none, and for 0/1 vectors with no common term the squared distance is just the sum of their lengths, so neighbourhoods are nearly random. My prediction of many small clusters at perplexity 30 was wrong.
 
UMAP (`reduce_dimensions_34`, plot: `chienhung_homework_plot_reduce_dimensions_34.png`) shows one mass with a few small groups at the edge. It looks less overlapped only because points are spread out; colours are interleaved everywhere. One group near (5.3, 12.5) looked purely neutral, but on the full-size plot it contains a positive point, so no group is class-pure.
 
When I asked how clustering could be improved, the agent suggested embeddings, SVD before projection, cosine distance and downsampling neutral. Embeddings fit my conclusion; SVD and the metric are not tool options; cosine would remove length but still gives 0 for posts without common words; downsampling only reduces over-plotting. Its claim that UMAP's local structure explains "single-coloured groups" is not supported.
 
## 7. Label binarization and cosine similarity
 
`binarize_labels_35` produced an 813 × 3 one-hot matrix in the order −1.0, 0.0, 1.0, with column sums 315 / 391 / 107 matching `describe_10`.
 
`cosine_similarity_tool` compares only one pair at a time, so instead of trying to describe all pairs, I used constructed sentences to test what the working matrix can distinguish:
 
| Result | Text 1 | Text 2 | Cosine |
|---|---|---|---|
| `cosine_sim_36` | the market is good | the market is not good | 1.0 |
| `cosine_sim_37` | your market is good | the market is good | 0.8165 |
 
A statement and its negation came out identical. The agent explained this by saying both sentences reduce to `market` only because `good` had also been removed; that seemed wrong, since `good` occurs in about 40 posts. I designed the second probe so that each explanation predicts a different value: 0.8165 if the tool uses `dtm_17` and keeps `good`, 0.707 if `good` is missing, 1.0 if a stop-word list is used, 0.75 if a new vocabulary is fitted. The result, 0.8165 = 2/√6, shows that text mode uses `dtm_17` and keeps `good`, so the agent was wrong. It also shows that `not` is absent from `dtm_17`, which can only be due to `max_df=0.1`: this confirms my suspicion from §3.
 
Two real same-class pairs, rows 0 and 4 (negative, `cosine_sim_38`) and rows 1 and 2 (neutral, `cosine_sim_39`), both give 0.0 with no empty vectors: the posts share no term. Two pairs are only an illustration, but they match the explanation of the t-SNE result in §6.
 
## 8. Generalization of my tools
 
All tools ran on the new data without code changes, but several defaults and parameter meanings only suited long, balanced newsgroup posts. `load_dataset_tool` silently picked the wrong of two label columns. `max_df=0.5` removed nothing, while `max_df=0.1` removed `not`. A variance threshold means "repeated within a post" on near-binary counts. `spearman_filter`'s default `max_terms=1500` dropped terms silently, and `min_sup` must be converted by hand for unequal classes. None is a bug, but each gave a plausible result to a different question, and the agent never flagged a default on its own.
 
Some design choices have side effects on this data. `keep=False` deletes every copy of a duplicate. Both DTM paths lose negation: `mine_patterns_tool` through its stop-word list, and `build_dtm_tool`, which has no stop-word option, through `max_df`. The correlation p-values are unreliable for rare terms and are reported without warning.
 
Several tools report too little to verify their output: top 20 only (`term_frequency_tool`, `mine_patterns_tool`); no loadings, metric, seed default or point-to-post mapping (`reduce_dimensions_tool`); no surviving terms and no texts (`cosine_similarity_tool`, which led the agent to guess wrongly); and `visualize_result_tool` reported 19 rows for a chart with 20 bars. Useful changes would be: requiring an explicit label column, exposing `keep`, a negation-preserving stop-word list, relative `min_sup`/`max_df` with a list of removed terms, a fixed default seed, and reporting the terms or posts behind each number.
 
## 9. Conclusions
 
In a bag-of-words representation of these posts, sentiment is barely visible. Correlations stay below 0.17, patterns are single words, no projection separates the classes, and same-class posts often share no term. The main cause is length: one-sentence posts give a sparse, nearly binary matrix in which most posts have nothing in common.
 
The labels follow topic and tone more than sentiment words: Trump, China and profanity mark negative posts, economic terms mark opinionated versus neutral ones, and the only robust positive marker is `good`, found through support rather than correlation. Together with the disagreeing label columns and the ambiguous examples in §2, this suggests the labels are noisy.
 
My preprocessing removed negation in every path, so "good" and "not good" are identical. Separating sentiment would need embeddings, which can bring posts without common words close, and preprocessing that keeps negation.
 
The pipeline from the newsgroups needed almost every parameter re-derived from this data's length, sparsity and class sizes. The most reliable checks were the ones I could derive myself (tied r values, the 0.09 variance bound, the cosine probes) rather than the agent's explanations, which several times stated guesses as facts. Listing parameters before each new tool and designing probes whose outcomes distinguish explanations caught the most problems.
