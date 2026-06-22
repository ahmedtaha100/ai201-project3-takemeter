"""
Single source of truth for the TakeMeter taxonomy.

Everything downstream (the Groq pre-labeler, the zero-shot baseline, the fine-tuning
notebook, and the deployed app) imports the label set and definitions from here, so the
definitions can never drift between components.

Community: r/sports COMMENTS. The subreddit's links and highlights are not the interesting
part. The varied discourse lives in the comments, where the same clip produces pure emotion,
bold opinions, and reasoned breakdowns, often a few replies apart.

Decision tree (apply in order):
  Q1. Does the comment make a debatable SPORTS claim / opinion / prediction / ranking?
      NO  -> reaction
  Q2. If YES, is the claim backed by a genuine explanatory/causal chain or multiple pieces
      of evidence (not a single stat used as garnish)?
      NO  -> hot_take
      YES -> analysis
"""

# Order is fixed: this is the integer-id order used by the model (id2label).
LABELS = ["reaction", "hot_take", "analysis"]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

DEFINITIONS = {
    "reaction": (
        "An emotional, in-the-moment expression responding to a play, game, or news item "
        "that makes NO substantive sports claim: hype, disbelief, humor, despair, or pure "
        "fandom."
    ),
    "hot_take": (
        "A strongly-stated, debatable sports opinion, claim, prediction, or ranking asserted "
        "with little or no supporting reasoning. A single stat dropped after the claim is "
        "garnish, not an argument, so it is still a hot take."
    ),
    "analysis": (
        "A reasoned, evidence-based sports argument that explains WHY: it lays out a "
        "causal/explanatory chain or stacks multiple pieces of evidence (stats, tactics, "
        "matchups, film, history)."
    ),
}

# Two clear examples + one hard edge case per label (used in planning.md / README / prompts).
EXAMPLES = {
    "reaction": ["WHAT A GOAL ARE YOU KIDDING ME", "bro I'm crying lmao we're so back"],
    "hot_take": ["He's washed, time to admit it", "She's already a top-5 athlete of all time and it's not close"],
    "analysis": [
        "They keep blitzing on third down, so until the back can pick it up the play-action "
        "never gets time to develop and the whole offense stalls out.",
        "His efficiency dropped because he's taking contested looks late in the clock now that "
        "the spacing around him collapsed, it's a scheme problem, not effort.",
    ],
}

BOUNDARY_RULES = (
    "Tiebreakers: (1) An emotional outburst that ALSO asserts a claim is hot_take, the claim "
    "is the labelable content, not the emotion. (2) A claim with a single stat tacked on is "
    "still hot_take; analysis needs an actual explanatory chain. (3) Sarcasm and jokes are "
    "labeled by the proposition they imply, not their surface form; a pure meme with no claim "
    "is reaction. (4) Awe plus a gaudy stat line plus emoji is reaction unless it states an "
    "explicit verbal ranking or claim ('he's the GOAT' becomes hot_take); meta-commentary "
    "mocking other takes or the discourse itself, with no on-field position, is reaction."
)


def definitions_block() -> str:
    """Render the label definitions + examples + rules as a prompt-ready block."""
    lines = []
    for label in LABELS:
        ex = "; ".join(f'"{e}"' for e in EXAMPLES[label][:2])
        lines.append(f"- {label}: {DEFINITIONS[label]} Examples: {ex}")
    lines.append(BOUNDARY_RULES)
    return "\n".join(lines)


def build_prompt(comment: str) -> str:
    """Prompt used by BOTH the Groq pre-labeler and the zero-shot baseline (same definitions)."""
    return (
        "You are classifying a single Reddit r/sports comment by the kind of discourse it is.\n\n"
        "Apply this decision tree:\n"
        "Q1. Does it make a debatable sports claim/opinion/prediction/ranking? "
        "If no -> reaction.\n"
        "Q2. If yes, is the claim backed by a real explanatory/causal chain or multiple "
        "pieces of evidence? If no -> hot_take. If yes -> analysis.\n\n"
        "Label definitions:\n"
        f"{definitions_block()}\n\n"
        f'Comment: """{comment}"""\n\n'
        "Output ONLY the label name, exactly one of: reaction, hot_take, analysis."
    )
