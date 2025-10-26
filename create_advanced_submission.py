"""
Create submission for advanced models:
- Domain Adaptation EEGNeX
- Cross-Task Pre-Training
- Hybrid CNN-Transformer-DA

Embeds model code into submission.py to avoid architecture mismatches
"""

import argparse
from pathlib import Path
import zipfile
import shutil


def get_model_files(model_type):
    """Get list of model files to include"""
    files = {
        'domain_adaptation': ['models/domain_adaptation_eegnex.py'],
        'cross_task': ['models/cross_task_pretrain.py'],
        'hybrid': ['models/hybrid_cnn_transformer_da.py'],
        'trial_level': ['models/trial_level_rt_predictor.py']
    }
    return files[model_type]


def get_model_import(model_type):
    """Get import statement for model"""
    imports = {
        'domain_adaptation': 'from domain_adaptation_eegnex import DomainAdaptationEEGNeX',
        'cross_task': 'from cross_task_pretrain import CrossTaskPretrainModel',
        'hybrid': 'from hybrid_cnn_transformer_da import HybridCNNTransformerDA',
        'trial_level': 'from trial_level_rt_predictor import TrialLevelRTPredictor'
    }
    return imports[model_type]


def get_model_creation_code(model_type, challenge):
    """Get model creation code"""
    if challenge == 'c1':
        output_range = '(0.5, 1.5)'
    else:
        output_range = '(-3, 3)'

    code = {
        'domain_adaptation': f'''DomainAdaptationEEGNeX(
            n_channels=129,
            n_times=900,
            challenge='{challenge}',
            num_subjects=100,
            output_range={output_range}
        )''',
        'cross_task': f'''CrossTaskPretrainModel(
            n_channels=129,
            n_times=900,
            num_tasks=6,
            task_names=['resting_state', 'video_watching', 'reading',
                       'contrast_change_detection', 'task_5', 'task_6'],
            output_ranges=[None, None, None, {output_range}, None, None]
        )''',
        'hybrid': f'''HybridCNNTransformerDA(
            n_channels=129,
            n_times=900,
            challenge='{challenge}',
            output_range={output_range},
            d_model=128,
            nhead=8,
            num_transformer_layers=4
        )''',
        'trial_level': f'''TrialLevelRTPredictor(
            n_channels=129,
            trial_length=200,
            pre_stim_points=50
        )'''
    }
    return code[model_type]


def get_model_forward_code(model_type):
    """Get forward pass code"""
    code = {
        'domain_adaptation': '''predictions = model(eeg_tensor)''',
        'cross_task': '''predictions = model(eeg_tensor, task_name='contrast_change_detection')''',
        'hybrid': '''predictions = model(eeg_tensor)''',
        'trial_level': '''predictions = model(eeg_tensor)'''
    }
    return code[model_type]


def create_submission(model_type, challenge, checkpoint_c1, checkpoint_c2, output_dir='submissions'):
    """
    Create submission zip file

    Args:
        model_type: 'domain_adaptation', 'cross_task', or 'hybrid'
        challenge: Name for submission (e.g., 'domain_adaptation_v1')
        checkpoint_c1: Path to C1 checkpoint
        checkpoint_c2: Path to C2 checkpoint
        output_dir: Output directory
    """
    print(f"Creating {model_type} submission: {challenge}")

    # Create temp directory
    temp_dir = Path('temp_submission')
    temp_dir.mkdir(exist_ok=True)

    # Copy model files
    model_files = get_model_files(model_type)
    for model_file in model_files:
        src = Path(model_file)
        dst = temp_dir / src.name
        shutil.copy(src, dst)
        print(f"  Copied {model_file}")

    # Copy checkpoints
    shutil.copy(checkpoint_c1, temp_dir / 'model_c1.pt')
    shutil.copy(checkpoint_c2, temp_dir / 'model_c2.pt')
    print(f"  Copied checkpoints")

    # Create submission.py
    model_import = get_model_import(model_type)
    model_creation_c1 = get_model_creation_code(model_type, 'c1')
    model_creation_c2 = get_model_creation_code(model_type, 'c2')
    forward_code = get_model_forward_code(model_type)

    submission_code = f'''"""
Submission script for {model_type}
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import model
{model_import}


def load_model_for_challenge(challenge_name, device='cuda'):
    """
    Load model for specific challenge

    Args:
        challenge_name: 'c1' or 'c2'
        device: 'cuda' or 'cpu'

    Returns:
        model: Loaded model
        metadata: Dict with normalization params
    """
    # Create model
    if challenge_name == 'c1':
        model = {model_creation_c1}
    else:
        model = {model_creation_c2}

    # Load checkpoint
    checkpoint_path = Path(__file__).parent / f'model_{{challenge_name}}.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Get normalization parameters
    metadata = {{
        'y_mean': checkpoint.get('y_mean', 0.0),
        'y_std': checkpoint.get('y_std', 1.0)
    }}

    return model, metadata


def predict(model, eeg_data, metadata, device='cuda'):
    """
    Make prediction

    Args:
        model: Loaded model
        eeg_data: EEG data (channels, time)
        metadata: Normalization parameters
        device: 'cuda' or 'cpu'

    Returns:
        prediction: Scalar prediction
    """
    # Convert to tensor
    eeg_tensor = torch.FloatTensor(eeg_data).unsqueeze(0).to(device)  # (1, channels, time)

    # Predict
    with torch.no_grad():
        {forward_code}

    # Get scalar prediction
    prediction = predictions.cpu().item()

    # Denormalize (model predicts normalized values)
    y_mean = metadata['y_mean']
    y_std = metadata['y_std']
    prediction_denorm = prediction * y_std + y_mean

    return prediction_denorm


if __name__ == '__main__':
    # Test submission
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Testing submission on {{device}}")

    # Test C1
    print("\\nTesting C1...")
    model_c1, metadata_c1 = load_model_for_challenge('c1', device=device)
    dummy_eeg = np.random.randn(129, 900).astype(np.float32)
    pred_c1 = predict(model_c1, dummy_eeg, metadata_c1, device=device)
    print(f"C1 prediction: {{pred_c1:.4f}}")

    # Test C2
    print("\\nTesting C2...")
    model_c2, metadata_c2 = load_model_for_challenge('c2', device=device)
    pred_c2 = predict(model_c2, dummy_eeg, metadata_c2, device=device)
    print(f"C2 prediction: {{pred_c2:.4f}}")

    print("\\nSubmission test passed!")
'''

    # Write submission.py
    submission_path = temp_dir / 'submission.py'
    with open(submission_path, 'w') as f:
        f.write(submission_code)
    print(f"  Created submission.py")

    # Create zip file
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    zip_path = output_dir / f'{challenge}_submission.zip'

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in temp_dir.iterdir():
            zipf.write(file, file.name)

    print(f"  Created {zip_path}")

    # Cleanup
    shutil.rmtree(temp_dir)

    print(f"\\n✓ Submission created: {zip_path}")
    print(f"  Ready to submit to competition!")

    return zip_path


def main():
    parser = argparse.ArgumentParser(description='Create Advanced Model Submission')
    parser.add_argument('--model', type=str, default='trial_level',
                       choices=['domain_adaptation', 'cross_task', 'hybrid', 'trial_level'],
                       help='Model type (default: trial_level)')
    parser.add_argument('--name', type=str, default='submission',
                       help='Submission name (default: submission)')
    parser.add_argument('--checkpoint_c1', type=str, required=True,
                       help='Path to C1 checkpoint')
    parser.add_argument('--checkpoint_c2', type=str, required=True,
                       help='Path to C2 checkpoint')
    parser.add_argument('--output_dir', type=str, default='submissions',
                       help='Output directory')

    args = parser.parse_args()

    create_submission(
        model_type=args.model,
        challenge=args.name,
        checkpoint_c1=args.checkpoint_c1,
        checkpoint_c2=args.checkpoint_c2,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
