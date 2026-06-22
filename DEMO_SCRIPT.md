# TakeMeter, Demo Video Script (3 to 5 minutes)

Legend: **bold = what you SAY out loud** · _italic = what you DO on screen_

Launch the app before you record: `python src/app.py`. It loads the fine-tuned model from
`models/distilbert-takemeter` and opens a local Gradio page. Have the evaluation report
(`evaluation_results.json` or README section 6) and `confusion_matrix.png` ready in a second tab
so you can flip to them at the end.

One honesty note for yourself before you start: the data is real r/sports COMMENTS from a public
Reddit archive (HuggingFaceGECLM/REDDIT_comments). I started on r/nba, but Reddit's API now returns
403 to unauthenticated clients, so live scraping died and I pulled real r/sports comments from the
archive instead. The app's input box still carries the original "r/nba comment" placeholder text from
when I built the UI. The comments themselves are real r/sports. Don't oversell, just say it plainly if
it comes up.

---

## 0:00 to 0:30, What it is

_Stay on the Gradio page so the title and the input box are visible._

**"This is TakeMeter. It takes a real sports comment off Reddit and sorts it into one of three kinds
of discourse. Reaction is pure emotion with no real claim. Hot take is a bold opinion thrown out with
no support behind it. And analysis is an argument that actually explains why. The hard part is that all
three come from the same fans arguing about the same games, so the model can't cheat by looking at the
topic. It has to pick up on whether there's a debatable claim, and whether that claim is actually
backed up."**

---

## 0:30 to 1:00, The taxonomy as a decision tree

_Optional: hold up or screen-share the two-question tree, or just talk through it._

**"The way I drew the line is a two-question tree. Question one: is there a debatable sports claim in
here at all? If no, it's a reaction, somebody just yelling. If yes, question two: is the claim backed
by a real explanatory chain or multiple pieces of evidence? If no, it's a hot take. If yes, it's
analysis. That second question is the whole ballgame, because that's the seam where a strong opinion
turns into an actual argument."**

---

## 1:00 to 2:15, Live classifications (label and confidence on screen)

_Paste each comment into the box and click Classify. Read the predicted label and the confidence bar
out loud each time. The app prints the top label, the confidence, and a bar chart for all three
classes, so the numbers are always visible._

_Comment 1 (warm-up reaction):_ paste `WHAT A SHOT ARE YOU KIDDING ME`

**"No claim, just adrenaline. It calls this a reaction, and it's confident. That's the easy end of the
scale."**

_Comment 2, the correct analysis example._ Paste this real test comment:

> The thing I don't like about this drill is most people have a strong side and a weak side for
> turning. And if you're right-handed as most people are (not necessarily right shooting), you usually
> turn left better. So McDavid has an advantage in this drill, assuming they're both right-handed. They
> should make them take one corner one way and the other corner the other way. You can see it in how
> Mackinnon doesn't do cross-overs in the turn. I bet if he were turning to his left he could get some
> cross-overs in.

**"This one is the model getting it right, and I want to sit on it for a second. The true label is
analysis and the fine-tuned model called it analysis at about 49 percent confidence. Listen to why it's
analysis and not just a hot take. There's a claim, McDavid has an advantage in this drill, and then
there's an actual chain behind it. Most people turn better to one side, here's the handedness reason,
here's the fix, and here's the evidence from watching Mackinnon's cross-overs. That 'so' and 'you can
see it in' is the explanatory chain. That's exactly the signal question two in the tree is asking
about, and the model picked it up."**

---

## 2:15 to 3:15, The miss that matters

_Comment 3, the real misclassification. Paste this:_

> The answer is pretty clear here...NFL RBs should stop riding elevators. Nothing but trouble in those
> tin cans.

**"Now here's one it gets wrong, and the way it's wrong is the most interesting thing in the whole
project. The true label is hot take. It's a bold, debatable claim, running backs should stop riding
elevators, and there's a little joke at the end but no real argument. The fine-tuned model called this a
reaction at 44 percent confidence. Look at the bars. Reaction 44, analysis 34, hot take only 21. It
didn't even put hot take in second place."**

**"And this isn't a one-off. The fine-tuned model never predicted hot take a single time on the whole
test set. Not once. Hot take is the smallest class and the most subjective one, and a 66 million
parameter model with only 176 training rows just collapsed it. Every real hot take got shoved into
reaction or analysis. So the confidence is low, the bars are flat, and on this comment it picked the
wrong neighbor."**

---

## 3:15 to 4:30, The evaluation report and the headline finding

_Flip to `evaluation_results.json` or README section 6, then to `confusion_matrix.png`._

**"Here's where it pays off to actually score this properly instead of trusting a vibe. The fine-tuned
DistilBERT got 73.7 percent accuracy but a macro-F1 of only 0.549. Macro-F1 averages the three classes
evenly, so it doesn't let a strong reaction score hide a dead class. And it exposes exactly that.
Reaction F1 is 0.82, analysis F1 is 0.83, and hot take F1 is 0.00, because the model never predicted it.
That single zero is dragging the whole macro number down, and accuracy alone would've hidden it."**

_Point at the confusion matrix, the hot_take row._

**"You can see it in the confusion matrix. The hot take row is five comments sent to reaction, three
sent to analysis, and zero staying home. Now compare it to the baseline. I ran a zero-shot Groq model,
llama-3.1-8b-instant, on the same test set. I picked a different model than the one I used to pre-label
the data on purpose, so this isn't circular. The zero-shot 8B got 81.6 percent accuracy and a macro-F1
of 0.779. It handled hot take at an F1 of 0.62, which is night and day from zero."**

**"So the headline is a little humbling and I think that's the point. The zero-shot 8B beat my fine-tuned
model by 0.230 macro-F1. The big pretrained model has seen enough language to recognize an unsupported
bold opinion. My small fine-tuned model, on 176 examples, couldn't carve out the smallest, fuzziest
class and just gave up on it. Fine-tuning isn't automatically the win. With a tiny, imbalanced dataset
on a genuinely subjective boundary, broad pretraining wins."**

---

## 4:30 to 5:00, Close

**"So, to recap. Real r/sports comments, a three-way discourse taxonomy built on a two-question decision
tree, a 66 million parameter DistilBERT fine-tuned locally, evaluated head to head against a zero-shot
8B baseline on the same held-out test set. The fine-tuned model is solid on reaction and analysis and
falls apart on hot take, the baseline beats it by 0.230 macro-F1, and the errors land right on the
subjective boundary the taxonomy flagged as hardest. That last part is what tells me the evaluation is
honest and not random. Thanks for watching."**

---

### Checklist before you hit record
- App launches and `models/distilbert-takemeter` loads ✓
- Both real examples paste cleanly: the "drill" analysis (correct, ~49%) and the "elevators" hot take
 (wrong, predicted reaction at 44%) ✓
- Label and confidence bars are on screen for every classification ✓
- `evaluation_results.json` / README section 6 and `confusion_matrix.png` open in a second tab ✓
- You can state the headline number from memory: zero-shot 8B beat fine-tuned by **0.230 macro-F1** ✓

**Total: about 4.5 minutes.** If you run long, trim the warm-up reaction in section 1:00 and go straight
to the two examples that carry the story.
