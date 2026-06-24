"""
Interactive XOR MLP playground.

Pick a backend (numpy from-scratch vs PyTorch autograd), tune hidden size /
learning rate / epochs, train, and see the loss curve + decision boundary.

Run: streamlit run app.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn

from data import get_xor_data
from mlp_numpy import NumpyMLP
from mlp_pytorch import TorchMLP

st.set_page_config(page_title="XOR MLP Playground", layout="centered")
st.title("XOR MLP Playground")
st.caption("See backprop train a tiny neural net to solve XOR — pick numpy (from scratch) or PyTorch (autograd).")

backend = st.sidebar.radio("Backend", ["numpy (from scratch)", "PyTorch (autograd)"])
hidden_size = st.sidebar.slider("Hidden neurons", min_value=2, max_value=16, value=4)
learning_rate = st.sidebar.slider("Learning rate", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
epochs = st.sidebar.slider("Epochs", min_value=100, max_value=10000, value=5000, step=100)

train_clicked = st.sidebar.button("Train", type="primary")

if train_clicked:
    X, y = get_xor_data()
    losses = []

    with st.spinner("Training..."):
        if backend.startswith("numpy"):
            model = NumpyMLP(input_size=2, hidden_size=hidden_size, output_size=1)
            for _ in range(epochs):
                losses.append(model.train_step(X, y, learning_rate))

            def predict(inputs):
                return model.forward(inputs)

        else:
            X_t = torch.tensor(X, dtype=torch.float32)
            y_t = torch.tensor(y, dtype=torch.float32)
            model = TorchMLP(input_size=2, hidden_size=hidden_size, output_size=1)
            loss_fn = nn.MSELoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

            for _ in range(epochs):
                prediction = model(X_t)
                loss = loss_fn(prediction, y_t)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            def predict(inputs):
                with torch.no_grad():
                    return model(torch.tensor(inputs, dtype=torch.float32)).numpy()

    st.subheader("Loss curve")
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (MSE)")
    st.pyplot(fig)
    st.metric("Final loss", f"{losses[-1]:.5f}")

    st.subheader("Final predictions")
    preds = predict(X)
    for inputs, target, pred in zip(X, y, preds):
        st.write(f"input = {inputs.tolist()} | target = {int(target[0])} | predicted = {float(pred[0]):.3f}")

    st.subheader("Decision boundary")
    st.caption("Color = network's prediction across the input space. Dots = the 4 actual XOR points.")
    grid_x, grid_y = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_preds = predict(grid_points).reshape(grid_x.shape)

    fig2, ax2 = plt.subplots()
    contour = ax2.contourf(grid_x, grid_y, grid_preds, levels=50, cmap="RdBu")
    ax2.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolor="black", s=120)
    fig2.colorbar(contour, ax=ax2, label="prediction")
    ax2.set_xlabel("input A")
    ax2.set_ylabel("input B")
    st.pyplot(fig2)

else:
    st.info("Set parameters in the sidebar, then click **Train**.")
