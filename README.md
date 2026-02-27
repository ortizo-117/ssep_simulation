# Tibial SSEP Simulation + Time–Frequency Analysis

Goal:This project simulates **tibial somatosensory evoked potentials (SSEPs)**, optionally preprocesses the simulated trials, and computes/visualizes **time–frequency representations** (e.g., power, ITPC) using a **family of Morlet wavelets**. From these, we generate image reports of ERP and ITPC features that are anonymized and saved for classification purposes. Finally, we evaluate the performance of different machine learning algorithms for classifying the SSEP conditions based on the generated images.
The objective is to test the ability of image-based simple classifiers to predict the condition and to compare the performance of different image reports (e.g., ERP only, TF only, combined) for this task. 



The main steps include:
1. Generate custom waveforms that mimic the characteristics of SSEPs, such as their frequency content, amplitude, and temporal structure.
2. Simulate the SSEP signals by adding noise and other artifacts to the generated waveforms
3. Apply various signal processing techniques to the simulated SSEP signals, such as filtering, averaging, and time-frequency analysis, to extract relevant features and analyze the results.
4. Generate visualizations and reports to summarize the evoked responses. 
5. Anonymize unlabel the responses and save them in JPEG format for classification purposes. 
6. Evaluate the performance of different algorithms for SSEP analysis, such as machine learning classifiers, and compare their results to the ground truth labels.
7. Document the code and results in a clear and organized manner, including explanations of the methods used and the assumptions made in the analysis.



---

## Project structure

- `01_main_simulation_script.ipynb`  
  Main simulation notebook: defines SSEP peak “ground truth” (e.g., normal/delayed/reduced conditions), generates multi-trial datasets, runs preprocessing, and produces plots/reports.

- `02_anonimizing_simulated_data.ipynb`  
  Utility notebook: copies `.jpg` outputs into a new folder while renaming them to random `subject_####.jpg` IDs and writing a CSV mapping.

- '03_evaluation_of_classification_algorithms.ipynb'  
  Main classification notebook: loads the generated '.jpg' files and trains/evaluates CNN classifiers to predict the original SSEP condition for a 3 class problem (normal/abnormal/abolished).

- `custom_classes.py`  
  Core implementation: peak generators, dataset simulator, preprocessing pipeline, wavelet-family builder, and TF metrics/plots. :contentReference[oaicite:1]{index=1}

---

## Requirements

Typical environment:
- Python 3.10+
- `numpy`, `scipy`, `matplotlib`, `pandas`

Install (example):
```bash
pip install numpy scipy matplotlib pandas
```

## Usage example for main simulation notebook for generation and visualization of simulated SSEP data
Here is a minimal code example that resembles the structure of the main_simulation_script.ipynb notebook, demonstrating how to generate simulated SSEP data, preprocess it, and visualize the results using a family of Morlet wavelets.

```python
from custom_classes import Peak, GaussianReboundPeak, Dataset, Preprocessor, wavelet_family
import numpy as np

# 1) Define peaks
peaks = [
    GaussianReboundPeak("P45", latency_ms=45, amp_uv=2, width_ms=5,
                        rebound_frac=1.5, rebound_delay_ms=7, rebound_width_ms=5),
    Peak("P75", latency_ms=75, amp_uv=2, width_ms=6),
]

# 2) Simulate dataset
ds = Dataset(
    peaks=peaks,
    fs=2000,
    tmin_ms=-50,
    tmax_ms=200,
    noise_rms_uv=0.5,
    rng_seed=0,
    generate=True,
    n_trials=300,
)

# 3) Preprocess (optional)
pp = Preprocessor(fs=ds.fs)
pp.add_bandpass(low_hz=1.0, high_hz=200.0, order=4)
ds.preprocess(pp)

# 4) Build wavelet family + compute TF
frex = np.linspace(4, 80, 30)
fwhm = np.linspace(0.4, 0.15, len(frex))  # example values (seconds)
wavtime = np.arange(-1, 1+1/ds.fs, 1/ds.fs)  # odd length recommended

wf = wavelet_family(frex=frex, fwhm=fwhm, srate=ds.fs, wavtime=wavtime, n_data=len(ds.trials[0][0]))
wf.build()

ds.compute_tf_complex_from_wavelet_family(wf)
metrics = ds.tf_metrics(baseline_ms=(-50, 0), kind="db")

# 5) Plot a combined TF+ERP report
ds.plot_report(tf_metric="itpc", tlim_ms=(-50, 200), flim_hz=(4, 80), show=True)
```
# Machine Learning Classification Description
The workflow consist of the following steps:
1. **Data Preparation**: Load the anonymized `.jpg` files containing the ERP and time-frequency representations of the simulated SSEP data. Each image corresponds to a specific SSEP condition.
2. **Label Mapping**: Use the CSV mapping file generated during the anonymization process to associate each image with its original SSEP condition (e.g., normal, abnormal, abolished).
3. **Image to Tensor Conversion**: Convert the loaded images into a format suitable for input into a Convolutional Neural Network (CNN). This involves resizing the images, normalizing pixel values, and converting them to tensors.
4. **Dataclass Definition**: Define a dataclass to represent the dataset, including attributes for the image data, labels, and any necessary metadata.
5. **Class imbalance Handling**: If the dataset is imbalanced (i.e., some classes have more samples than others), apply techniques such as oversampling, undersampling, or class weighting to ensure that the model learns effectively from all classes.
4. **Defining dataloaders**: Create dataloaders for training and validation sets to efficiently feed data into the CNN during the training process.
5. **Model Definition**: Defined 3 different CNN architectures - Simple CNN, ResNet, and EfficientNet - to classify the SSEP conditions based on the input images. ResNet and EfficientNet backbone architectures were frozen to leverage pre-trained weights, while the final classification layers were trained on the SSEP dataset.
6. **Two input late fusion model**: In addition to the individual CNN models, a late fusion model was implemented that takes both ERP and time-frequency images as input. This model combines the features extracted from both types of images to make a final classification decision.
7. **Training and Evaluation definitions: Define the training loop, loss function, and evaluation metrics (e.g., accuracy, precision, recall) to train the CNN models and evaluate their performance on a validation set.
8. **Experimental setup**: Train each of the defined CNN models on the training dataset and evaluate their performance on the validation set. Compare the results to determine which model architecture performs best for classifying the SSEP conditions based on the generated images.
