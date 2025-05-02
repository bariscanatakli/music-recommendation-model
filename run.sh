#!/bin/bash
# Çalıştırma örneği

# Sanal ortamı etkinleştir
source venv/bin/activate

# Modeli eğit
python train.py --model improved --epochs 30 --batch_size 32

# Tahmin örneği
python predict.py --model_path outputs/models/music_genre_classifier_improved.h5 --audio_file sample.mp3

echo "İşlem tamamlandı!"