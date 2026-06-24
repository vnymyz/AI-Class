# Session 1 — MLP from Scratch (XOR)

See `session1_theory.md` for the full theory writeup with diagrams.
See `ROADMAP.md` for the full multi-session deep learning curriculum.

## Structure

```
10.another-deep-learning/
├── ROADMAP.md          full curriculum (session 1-10)
├── session1_theory.md  theory writeup with ASCII diagrams (this session)
├── requirements.txt
├── dataset/            (empty for now — real datasets go here in later sessions)
├── src/
│   ├── data.py          XOR dataset (4 rows, classic non-linear problem)
│   ├── mlp_numpy.py      MLP built fully by hand: forward, backward (chain rule), updates
│   ├── mlp_pytorch.py    same architecture, defined with nn.Module
│   └── app.py            Streamlit playground: tune params, train live, see decision boundary
└── notebooks/
    └── notebook.ipynb    step-by-step walkthrough: theory notes + train both models + plots
```

`src/data.py`, `src/mlp_numpy.py`, `src/mlp_pytorch.py` are plain importable modules — both the
notebook and the Streamlit app reuse them, so the model logic only lives in one place.

## Run

```
pip install -r requirements.txt

# step-by-step notebook (theory + training + plots)
jupyter notebook notebooks/notebook.ipynb

# interactive playground (sliders, live decision boundary)
streamlit run src/app.py
```

Both backends should drive loss near 0 and correctly predict all 4 XOR cases.

## What to compare (numpy vs PyTorch)

| Step              | numpy version (`mlp_numpy.py`)      | PyTorch version (`mlp_pytorch.py`) |
|-------------------|--------------------------------------|----------------------------------------------------------|
| Forward pass      | written by hand (`forward`)          | `model(X)`                                                |
| Loss              | written by hand (`compute_loss`)     | `nn.MSELoss()`                                             |
| Backward pass     | written by hand (`backward`, chain rule) | `loss.backward()` (autograd)                           |
| Weight update     | written by hand (`-= learning_rate * grad`) | `optimizer.step()`                                  |

Same math, same result — PyTorch just automates steps 3 and 4.

## Learning path (tutor view)

Don't jump straight to code. Order matters — each step builds on the last.

**Step 1 — Read theory first, slow.**
Open `session1_theory.md`. Don't skim. Sections 1-3 (neuron, activation, stacking into a network)
are the "what". Sections 4-6 (loss, gradient descent, backprop) are the "how it learns" — this is
the part most people gloss over and then get lost later. Sit with the hill diagram (section 5)
until "gradient = slope, walk downhill" clicks. Everything in deep learning builds on that one idea.

**Step 2 — Read `src/mlp_numpy.py` next to the theory.**
Don't just run it. Map each line to a concept:
- `forward()` = section 3 (stacking neurons forward)
- `compute_loss()` = section 4
- `backward()` = section 6 (this is the chain rule, written out by hand — go slow here)
- `train_step()` = section 7 (the full loop)
This file is the most important one in this session. If you understand every line here, you
understand backprop — not just "know the word", actually understand it.

**Step 3 — Run `notebooks/notebook.ipynb` cell by cell.**
Watch the loss number drop each print. Pause and predict: "will loss go down or up next?" before
running the next cell. This builds intuition for what training "feels like" when it's working
vs stuck.

**Step 4 — Read `src/mlp_pytorch.py` right after.**
Same architecture, but no `backward()` method anywhere — `loss.backward()` does it. Compare
side by side with `mlp_numpy.py`. This is the moment you'll realize frameworks aren't magic,
they're just automating step 2.

**Step 5 — Play with `src/app.py` (`streamlit run src/app.py`) last, after you understand why.**
Now experiment freely:
- Drop hidden neurons to 1 — watch it fail to learn XOR (proves why hidden layers matter).
- Crank learning rate to 2.0 — watch loss curve get noisy/explode (proves why learning rate matters).
- Switch numpy vs PyTorch backend — confirm both reach the same decision boundary.
Tweaking without step 1-4 first just teaches you "what slider does what", not why. Theory first,
then experiment to confirm the theory — not the other way around.

**Self-check before moving to Session 2 (CNN):** can you explain backprop out loud, in plain words,
without saying "magic" or "it just works"? If yes, you're ready. If not, re-read `mlp_numpy.py`'s
`backward()` method one more time — that's the whole trick, it's only ~10 lines.
