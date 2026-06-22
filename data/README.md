# data/

## Source

The community here is r/sports comments. The original plan was to pull from r/nba, but
Reddit's API started returning HTTP 403 to unauthenticated clients, so live scraping was
a dead end. The pivot: real r/sports comments from a public Reddit archive on Hugging Face,
`HuggingFaceGECLM/REDDIT_comments` (the r/sports split, PushShift data spanning 2006 to 2023).
The comments themselves are real Reddit text. Only the collection channel changed, from a live
API to an archived dump.

## Files

- `raw_comments.csv`: 900 raw comments streamed from the r/sports split (output of `src/collect.py`).
- `takemeter_labeled.csv`: the 252 labeled comments. Columns are `text,label,notes`. This is the
  submission dataset.
- `needs_review.csv`: a 38-row queue of low and medium confidence rows flagged for the human
  review pass.

## Taxonomy

Each comment gets one of three discourse labels:

- `reaction`: a response with no debatable sports claim behind it (cheering, a one-liner, an emoji).
- `hot_take`: a debatable sports claim stated without any real backing.
- `analysis`: a debatable claim that comes with an explanatory chain or multiple pieces of evidence.

## Label distribution

The final 252 comments break down like this:

| label    | count | share  |
|----------|-------|--------|
| reaction |   115 | 45.6%  |
| hot_take |    52 | 20.6%  |
| analysis |    85 | 33.7%  |

Every class clears 20% and none of them sits above 70%, so the set stays reasonably balanced.
