# Labs-CV
### Repositorio para los laboratorios y proyectos del grupo.

### Integrantes: 
 - Adrian Fulladolsa
 - Renatto Guzmán

### Laboratorio 4 U-Net:


### Comandos
Generacion de Dataset

    # Use all defaults (32x32 windows, 500k samples, bsds500_data directory)
    python lab4_1_DataSet.py

    # Custom window size and sample count
    python lab4_1_DataSet.py --window_size 16 --num_samples 1000000

    # Custom directory and all parameters
    python lab4_1_DataSet.py --base_dir my_dataset --window_size 64 --num_samples 250000

    # Different train/test splits
    python lab4_1_DataSet.py --test_size 0.15 --val_size 0.15

    # Skip steps if you already have data
    python lab4_1_DataSet.py --skip_download --window_size 16
    python lab4_1_DataSet.py --skip_filtering --num_samples 100000

Entrenamiento de modelo

    python unet_trainer.py --dataset_dir bsds500_data/dataset --action train --speed_mode

    # o
    
    python unet_trainer.py \
    --dataset_dir bsds500_data/dataset \
    --action train \
    --aggressive_mode \
    --separable_conv \
    --reduced_data 0.1 \
    --epochs 20 \
    --batch_size 128

Evaluacion de mejor modelo

    python unet_trainer.py --dataset_dir bsds500_data/dataset --action evaluate

Inferencia de imagen con mejor modelo

    python unet_trainer.py \
    --action infer \
    --input_image test_image.jpg \
    --output_image filtered_result.jpg \
    --model_path models/best_model.h5 \
    --window_size 32 \
    --stride 16



Mejores resultados

    Test Results:
    Loss: 0.000748
    MAE: 0.016517
    MSE: 0.000748
    PSNR: 34.05 dB

