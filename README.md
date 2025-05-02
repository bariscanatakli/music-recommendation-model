# Music Genre Classification and Recommendation System

A deep learning-based system for music genre classification and recommendation using spectrogram analysis and convolutional neural networks.

## Features

- Audio to spectrogram conversion
- Multiple model architectures:
  - ResNet-inspired CNN
  - Efficient Mobile-inspired model
  - Ensemble model with multiple pathways
  - Transformer-hybrid architecture
  - Improved model with squeeze-excitation blocks
- Training pipeline with data augmentation
- Performance visualization and analysis

## Requirements

- Python 3.8+
- TensorFlow 2.12+
- Other dependencies listed in requirements.txt

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Music-Recommendation-Using-Deep-Learning.git
cd Music-Recommendation-Using-Deep-Learning

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your audio files in the `data/raw` directory and run the preprocessing:

```bash
python -m data.preprocess --input data/raw --output data/processed
```

### Training

To train a model:

```bash
python train.py --model improved --epochs 30 --batch_size 32
```

Available models: `improved`, `ensemble`, `resnet`, `efficient`, `transformer`

### Prediction

To classify music tracks:

```bash
python predict.py --model_path model_checkpoints/best_model.h5 --audio_file path/to/song.mp3
```

## Project Structure

- `models/`: Neural network model architectures
- `data/`: Data loading and preprocessing utilities
- `train.py`: Model training script
- `predict.py`: Music genre prediction script
- `utils.py`: Helper functions

## Results

The improved model achieves 92% accuracy on the test dataset. Confusion matrices and performance metrics are generated during training.

## License

MIT

## Citation

If you use this code in your research, please cite:

```
@software{MusicRecommendation2025,
  author = {Your Name},
  title = {Music Genre Classification and Recommendation Using Deep Learning},
  year = {2025},
  url = {https://github.com/yourusername/Music-Recommendation-Using-Deep-Learning}
}
```