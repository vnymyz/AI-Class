# Session 1 Theory: How a Neural Network Actually Learns

Goal today: understand backprop + build tiny MLP (Multi-Layer Perceptron) from scratch.

No heavy math jargon here — just the idea, in plain words + pictures.

---

## 1. What is a Neuron?

Think of one neuron like a tiny voting machine.

```
x1 ----w1----\
              \
x2 ----w2------>  [ sum + bias ] --> [ squash ] --> output
              /
x3 ----w3----/
```

- `x1, x2, x3` = inputs (numbers coming in)
- `w1, w2, w3` = weights = "how much do I trust this input"
- `bias` = a knob to shift the result up/down
- `squash` = activation function (explained below)

That's it. One neuron = weighted vote + a twist.

---

## 2. Why do we "squash" the result? (Activation Function)

Without squashing, stacking many neurons together is mathematically the same as having just ONE neuron. No matter how many layers, it stays a straight line — can't learn curvy/complex patterns (like XOR, images, language).

The squash function adds a "bend", so the network CAN learn complex shapes.

Common squash functions:

```
Sigmoid                 ReLU                    Softmax
   1 |     ____            |    /              turns numbers into
     |    /                |   /               probabilities that
   0 |___/                 |__/____             sum to 100%
     -5  0   5              0                  e.g. [cat 80%, dog 20%]
(squishes to 0~1)     (0 if negative,
                        else pass through)
```

- **Sigmoid**: good for "yes/no" style output (0 to 1)
- **ReLU**: most common in hidden layers, simple & fast: "if negative, kill it; else, keep it"
- **Softmax**: used at the very end for multi-class problems (cat/dog/bird...)

---

## 3. Stacking Neurons = a Network (MLP)

```
 INPUT LAYER      HIDDEN LAYER       OUTPUT LAYER

   (x1) ---\    /---(h1)---\
            \  /            \
   (x2) -----><----(h2)------>---- (output)
            /  \            /
   (x3) ---/    \---(h3)---/

   Data goes in -->  gets transformed -->  prediction comes out
```

This left-to-right pass = **Forward Pass**. Just plugging numbers through the machine to get a guess.

---

## 4. How do we know if the guess is good or bad? (Loss Function)

After forward pass, we get a prediction. Compare it to the real answer.

```
Prediction: 0.9 (means "90% sure it's a cat")
Real answer: 1.0 (it WAS a cat)

Loss = how far off we are (small loss = good guess, big loss = bad guess)
```

Loss is just **one number** that says "how wrong was I". Goal of training = make this number smaller and smaller over time.

---

## 5. How does the network improve? (Gradient Descent)

Imagine loss as hills and valleys. We are somewhere on this landscape, and we want to walk down to the lowest point (smallest error).

```
Loss
  |     *  <- start here (bad, high loss)
  |      \
  |       \
  |        \
  |         \___
  |             \___
  |                 \___*  <- goal (low loss)
  |____________________________ weight value
```

- The "slope" at our current position = **gradient**
- We take a small step downhill (opposite of slope direction)
- Step size = **learning rate** (too big = overshoot, too small = super slow)

We repeat this many times (many small steps) until loss gets low.

---

## 6. How do we know which direction is "downhill" for EVERY weight? (Backpropagation)

This is the key trick that makes deep learning possible.

```
FORWARD  (left to right): make a guess
   input -> hidden -> output -> [compare to real answer] -> LOSS

BACKWARD (right to left): figure out blame
   LOSS -> output -> hidden -> input
   "how much did YOU contribute to this error?" (asked to every weight)
```

Backprop = passing the "blame" for the error backward through the network, layer by layer, so every single weight learns exactly how much to adjust (and in which direction).

Under the hood it's just **chain rule** from calculus, applied automatically layer by layer. You don't need to be a calculus expert — just understand the idea: **error flows backward, weights get nudged, repeat.**

---

## 7. The Full Training Loop

```
 ┌─────────────────────────────────────────────┐
 │  1. Forward pass  -> get a prediction        │
 │  2. Compute loss  -> how wrong was it?       │
 │  3. Backward pass -> figure out blame        │
 │  4. Update weights -> nudge in right         │
 │     direction (small step)                  │
 │  5. Repeat thousands of times                │
 └─────────────────────────────────────────────┘
        loss should keep going down over time
```

---

## Today's Mini Project

We'll build a tiny MLP that learns **XOR**:

```
Input A | Input B | XOR Output
   0    |    0    |    0
   0    |    1    |    1
   1    |    0    |    1
   1    |    1    |    0
```

Why XOR? It's the classic example that a single neuron CANNOT solve (not a straight-line-separable problem) — proving why we need hidden layers + activation functions.

Steps:
1. Build the network from scratch in plain numpy (write forward pass, backward pass, weight updates ourselves — no shortcuts).
2. Watch the loss number go down as it trains.
3. Rebuild the exact same thing in PyTorch (a few lines) to see what the framework automates for us (autograd = automatic backprop).

