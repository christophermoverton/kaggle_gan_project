


#  Monet Style Transfer with CycleGAN | Kaggle GAN Project

This project explores unpaired image-to-image translation using CycleGAN, with the goal of transforming real-world photographs into Monet-style paintings. It was developed as part of a deep learning coursework assignment and trained on data from Kaggle’s “I’m Something of a Painter Myself” competition.

---

##  Project Structure

- `Week5DL_project.ipynb` – The main training notebook containing model setup, loss definitions, training loop, and visualizations.
- `outputs/` – Folder for storing generated images (e.g., `epoch_*.png`) and zipped outputs.
- `saved_models/` – Directory to store saved generator weights.
- `images.zip` – Final zipped archive of Monet-style generated images (for submission or inference).

---

## 🚀 Key Features

- ✅ **CycleGAN** for unpaired image translation
- ✅ **Identity, Cycle Consistency, Perceptual, and Edge Preservation Losses**
- ✅ Supports **image export in `.jpg` format** and automatic zipping
- ✅ Visualizes training losses and generated outputs
- ✅ Configured for Kaggle GPU environments (single or dual GPU)

---

##  Training Details

- **Input size:** 256x256 RGB
- **Dataset:** [Kaggle Painter Competition](https://www.kaggle.com/competitions/gan-getting-started/data)
- **Batch size:** Configurable
- **Steps per epoch:** Automatically calculated from dataset size
- **Output images:** 7,000–10,000 Monet-style `.jpg`s zipped into `images.zip`

---

## Loss Functions

The model is trained with a composite generator loss function, including:

- **Adversarial Loss** (Binary Crossentropy)
- **Cycle Consistency Loss** (L1)
- **Identity Loss** (L1)
- **Perceptual Loss** (VGG19 feature-space loss)
- **Edge Loss** (Sobel + brightness-weighted directional difference)

Each component is weighted and tracked across epochs for balanced optimization.

---

##  Inference / Image Generation

To generate Monet-style images after training:

```python
sample_photo = load_image_for_inference("your_photo.jpg")
generate_and_show(sample_photo, G_photo_to_monet, title="Photo → Monet")
```

To generate a full batch and zip:

```python
generate_and_save_images(G_photo_to_monet, photo_ds, total_count=7000)
zip_images_folder("monet_jpg_outputs", "images.zip")
```

---

##  Saving & Reloading Models

```python
G_photo_to_monet.save("saved_models/G_photo_to_monet.keras")
G_photo_to_monet = tf.keras.models.load_model("saved_models/G_photo_to_monet.keras")
```

---

## ⚙️ Environment Setup

This project was developed and tested on a **Linux environment via Windows Subsystem for Linux (WSL)** — specifically **WSL2**, which allows for a seamless development experience using Linux tools on a Windows machine.

Python dependencies are managed using `pip`, and a full list of required packages is included in the `requirements.txt` file.

---

### ✅ To Set Up the Environment:

1. **Clone the repository:**

```bash
git clone https://github.com/christophermoverton/kaggle_gan_project.git
cd kaggle_gan_project
```

2. **Create a virtual environment (recommended):**

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows (WSL): source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **(Optional) Verify TensorFlow GPU availability:**

```python
import tensorflow as tf
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
```

---

### 🧪 Notes for WSL Users:

- This project was developed using **WSL2 with Ubuntu 22.04**.
- Ensure you have WSL configured with GPU passthrough enabled if training with TensorFlow GPU (see [Microsoft’s WSL GPU guide](https://learn.microsoft.com/en-us/windows/ai/directml/dml-provider) for setup).
- GPU support for TensorFlow under WSL requires compatible NVIDIA drivers (`nvidia-smi` should work inside your WSL terminal).


---

##  Environment

You can export the current environment for reproducibility:

```bash
pip freeze > requirements.txt
```

Or convert to a conda-compatible YAML manually.

---

##  Credits

Project by [Christopher Overton](https://github.com/christophermoverton)  
Based on CycleGAN architecture and adapted for Kaggle competition workflows.

---

##  License

This project is for academic purposes and adheres to fair use of Kaggle competition datasets.
```

