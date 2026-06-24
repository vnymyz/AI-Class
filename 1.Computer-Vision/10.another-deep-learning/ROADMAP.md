# Deep Learning Deep-Dive Roadmap

Goal: cover popular DL/NN architectures, theory + hands-on mini project each.
Already done before this roadmap: CNN image classification (Animal-10), YOLOv8n object detection, MediaPipe hand tracking.

## Session 1 — Foundations: Backprop + MLP (today)
Theory: perceptron, activation funcs (sigmoid/ReLU/softmax), forward pass, loss funcs, chain rule, gradient descent, backprop.
Mini project: MLP from scratch in raw numpy (no framework) on XOR / small tabular dataset. Then same net rebuilt in PyTorch to compare.

## Session 2 — CNN internals (you used CNNs, now the "why")
Theory: convolution math, kernels/filters, padding/stride, pooling, feature maps, receptive field.
Mini project: build small CNN by hand (numpy conv layer) on tiny image set, then compare to your Animal-10 PyTorch model.

## Session 3 — RNN / LSTM / GRU
Theory: sequence modeling, hidden state, vanishing gradient problem, LSTM gates, GRU simplification.
Mini project: text classification (reuse your sentiment data) with LSTM, compare vs your earlier non-DL NLP approach.

## Session 4 — Attention + Transformer
Theory: self-attention, multi-head attention, positional encoding, encoder/decoder.
Mini project: mini transformer text classifier from scratch (attention block by hand), then HuggingFace version side by side.

## Session 5 — Autoencoders + VAE
Theory: encoder/decoder, latent space, reconstruction loss, VAE = + probabilistic latent space.
Mini project: autoencoder for image denoising / anomaly detection.

## Session 6 — GAN
Theory: generator vs discriminator, adversarial loss, mode collapse.
Mini project: small GAN generating simple images (e.g. MNIST digits).

## Session 7 — Diffusion models
Theory: forward noising process, reverse denoising, score matching (conceptual level, not full math).
Mini project: tiny diffusion model on simple 2D data or small images.

## Session 8 — Graph Neural Networks (GNN)
Theory: message passing, node/edge embeddings, when graphs beat grids/sequences.
Mini project: node classification on small graph dataset.

## Session 9 — Deep RL
Theory: reward, policy, value function, Q-learning → DQN → policy gradient.
Mini project: DQN agent on simple Gym environment (e.g. CartPole).

## Session 10 — Wrap-up: combining architectures
Theory: how YOLO (CNN+head), ViT (Transformer for images), Siamese nets, etc combine fundamentals.
Mini project: pick one hybrid architecture, build/finetune small version.

---
Order can shift based on interest. Each session: theory first (analogies before math/jargon), confirm before big code, then hands-on mini project.
