# Animal Image Classifier

Image classification project that recognizes 10 animal classes using transfer
learning (MobileNetV2) on the [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animal10)
dataset, with a Streamlit app for interactive predictions.

## Approach

- **Deep Learning** → **CNN (Convolutional Neural Network)** → **Transfer
  Learning (MobileNetV2)** → **Supervised Image Classification**.
- MobileNetV2 was pretrained on ImageNet (millions of images). We reuse its
  learned features and only train a new classification head on top for our
  10 animal classes, instead of training a CNN from scratch.

## Classes

`butterfly, cat, chicken, cow, dog, elephant, horse, sheep, spider, squirrel`

> The model can only predict these 10 classes. An unseen animal (e.g. tiger)
> will be mapped to the visually closest class (e.g. cat).

## How this project works

1. **Data**: ~26K labeled animal photos (10 classes) split into training
   (80%) and validation (20%) sets.
2. **Training**: the model looks at training images repeatedly, adjusting
   itself to get better at predicting the correct animal. After each pass,
   it's also tested on the validation set (images it didn't train on) to
   check if it's actually learning general patterns, not just memorizing.
3. **Saving**: the trained model is saved to `models/animal_classifier.keras`.
4. **Inference (the app)**: `app/app.py` loads that saved model. You upload
   any photo → it gets resized/preprocessed the same way as training images
   → the model outputs a probability for each of the 10 classes → the app
   shows the top prediction + a confidence breakdown.

> Note: the model can only output one of the 10 trained classes. An animal
> outside that list (e.g. tiger) gets mapped to the closest-looking class
> (e.g. cat) — it's not "wrong", just outside the model's vocabulary.

## Key concepts (training)

Think of training like studying flashcards for an exam:

- **Epoch**: one full pass through all the training images. Epoch 1 = first
  time through the deck, epoch 2 = going through it again, etc.
- **Loss**: a "how wrong was I" score on the data the model is training on.
  Lower = fewer mistakes = better. Naturally drops as training continues.
- **val_loss / val_accuracy**: the same scores, but measured on the
  validation set — images the model never trains on. This is the "practice
  exam" that shows whether the model truly understood the patterns, not just
  memorized the training images.
- **Plateau**: the point where val_loss stops improving no matter how many
  more epochs you run. The model has learned what it can from this data.
- **Overfitting**: training past the plateau — train loss keeps dropping
  (memorizing training images) while val_loss gets *worse* (it generalizes
  worse to new images). In our run, this happened after epoch 4.
- **Early stopping**: a training rule that stops automatically once val_loss
  stops improving for a few epochs, and keeps the best-performing version of
  the model — avoids overfitting and guessing how many epochs to use upfront.

## Project structure

```
7.image-classification/
├── data/
│   ├── raw/              # dataset (gitignored)
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb       # class balance, sample images, image size distribution
│   └── 02_training.ipynb  # transfer learning training, evaluation, save model
├── src/
│   ├── data_loader.py     # tf.data dataset loading + augmentation
│   ├── model.py            # MobileNetV2 model build/load
│   └── utils.py            # shared constants, image preprocessing
├── models/
│   └── animal_classifier.keras   # trained model (gitignored)
├── app/
│   └── app.py              # Streamlit demo app
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

### Get the dataset

1. Create a Kaggle account and generate an API token (Settings → API).
2. Save the token to `~/.kaggle/`.
3. Download via `notebooks/01_eda.ipynb` (first cell), or manually from
   [kaggle.com/datasets/alessiocorrado99/animal10](https://www.kaggle.com/datasets/alessiocorrado99/animal10)
   and extract into `data/raw/`.

## Usage

1. **EDA**: run `notebooks/01_eda.ipynb` to explore class balance, sample
   images, and image size distribution.
2. **Training**: run `notebooks/02_training.ipynb` to train MobileNetV2
   (frozen base, transfer learning) and save the model to `models/`.
3. **App**: run the Streamlit demo from the project root:

   ```bash
   python -m streamlit run app/app.py
   ```

   Upload an image and the app shows the predicted class with a confidence
   breakdown for all 10 classes.

## Current results

- Trained 10 epochs, frozen MobileNetV2 base + dense classification head.
- Best epoch: 4 (`val_accuracy` ≈ 95.6%, `val_loss` ≈ 0.158).
- Saved model: epoch 10 (`val_accuracy` ≈ 94.7%) — slightly overfit past
  epoch 4, but still good for a demo.

## Possible improvements

- **Early stopping**: add `EarlyStopping(monitor="val_loss", patience=3,
restore_best_weights=True)` to `model.fit` so training auto-stops at the
  best epoch instead of overfitting past it.
- **Fine-tuning**: unfreeze the top layers of MobileNetV2 (low learning rate)
  for a small accuracy boost — only worth it after early stopping is in
  place, since the current model hasn't converged at its best weights yet.
- **Class imbalance**: dataset ranges from ~1.4K (elephant) to ~4.9K (dog)
  images per class — consider `class_weight` in `model.fit` to balance this.
- **Confusion matrix**: add an evaluation cell in the training notebook to
  see which classes get confused with each other (e.g. cat vs dog vs spider
  vs butterfly).
- **Out-of-distribution handling**: add a confidence threshold in the app
  (e.g. "uncertain" message if top confidence < 50%) so unfamiliar animals
  don't get a misleadingly confident label.
- **Deployment**: containerize with Docker, or deploy to Streamlit Community
  Cloud / Hugging Face Spaces.
