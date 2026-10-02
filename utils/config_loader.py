import configparser
import os
import argparse


def load_config(config_path):
    """
    Load configuration from .conf file.
    """
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    config.read(config_path)
    return config


def config_to_args(config):
    """
    Convert config to argument namespace.
    """
    parser = argparse.ArgumentParser()
    
    if 'Paths' in config:
        parser.add_argument('--image_dir', type=str, default=config.get('Paths', 'image_dir'))
        parser.add_argument('--text_dir', type=str, default=config.get('Paths', 'text_dir'))
        parser.add_argument('--taxi_flow_path', type=str, default=config.get('Paths', 'taxi_flow_path'))
        parser.add_argument('--geo_path', type=str, default=config.get('Paths', 'geo_path'))
        parser.add_argument('--save_dir', type=str, default=config.get('Paths', 'save_dir_stage1'))
        if config.has_option('Paths', 'save_dir_stage2'):
            parser.add_argument('--save_dir_stage2', type=str, default=config.get('Paths', 'save_dir_stage2'))
        else:
            parser.add_argument('--save_dir_stage2', type=str, default=config.get('Paths', 'save_dir_stage1'))
        parser.add_argument('--log_dir', type=str, default=config.get('Paths', 'log_dir'))
        parser.add_argument('--result_dir', type=str, default=config.get('Paths', 'result_dir'))
    
    if 'Data' in config:
        parser.add_argument('--data', type=str, default=config.get('Data', 'data'))
        parser.add_argument('--num_nodes', type=int, default=config.getint('Data', 'num_nodes'))
        parser.add_argument('--train_ratio', type=float, default=0.8, help='Train set ratio')
        parser.add_argument('--val_ratio', type=float, default=0.1, help='Val set ratio')

    if 'Model' in config:
        parser.add_argument('--his_len', type=int, default=config.getint('Model', 'his_len'))
        parser.add_argument('--pred_len', type=int, default=config.getint('Model', 'pred_len'))
        parser.add_argument('--input_dim', type=int,
                  default=config.getint('Model', 'input_dim', fallback=2))
        parser.add_argument('--output_dim', type=int,
                  default=config.getint('Model', 'output_dim', fallback=2))
        parser.add_argument('--d_model', type=int, default=config.getint('Model', 'd_model'))
        parser.add_argument('--hidden_dim', type=int, default=config.getint('Model', 'hidden_dim'))
        parser.add_argument('--num_experts', type=int, default=config.getint('Model', 'num_experts'))
        parser.add_argument('--feature_dim', type=int, default=config.getint('Model', 'feature_dim'))
        parser.add_argument('--prompt_dim', type=int,
                  default=config.getint('Model', 'prompt_dim', fallback=config.getint('Model', 'hidden_dim')))
        parser.add_argument('--image_token_count', type=int,
                  default=config.getint('Model', 'image_token_count', fallback=8))
        parser.add_argument('--use_remoteclip', action='store_true', 
                          default=config.getboolean('Model', 'use_remoteclip'))
        parser.add_argument('--remoteclip_model', type=str, 
                          default=config.get('Model', 'remoteclip_model'))
        parser.add_argument('--use_glam', action='store_true',
                  default=config.getboolean('Model', 'use_glam', fallback=True))
        parser.add_argument('--glam_knn_k', type=int,
                  default=config.getint('Model', 'glam_knn_k', fallback=8))
        parser.add_argument('--prototype_tau', type=float,
                  default=config.getfloat('Model', 'prototype_tau', fallback=0.2))
        parser.add_argument('--use_prototype_router', action='store_true',
                  default=config.getboolean('Model', 'use_prototype_router', fallback=True))
        parser.add_argument('--dropout', type=float, default=config.getfloat('Model', 'dropout'))
        if config.has_option('Model', 'mask_flow_input'):
            parser.add_argument('--mask_flow_input', action='store_true',
                              default=config.getboolean('Model', 'mask_flow_input'))
        else:
            parser.add_argument('--mask_flow_input', action='store_true', default=False)
    
    if 'Training' in config:
        parser.add_argument('--batch_size', type=int, default=config.getint('Training', 'batch_size'))
        parser.add_argument('--epochs', type=int, default=config.getint('Training', 'epochs'))
        parser.add_argument('--lr', type=float, default=config.getfloat('Training', 'lr'))
        parser.add_argument('--weight_decay', type=float, 
                          default=config.getfloat('Training', 'weight_decay'))
        parser.add_argument('--lambda_bal', type=float, 
                          default=config.getfloat('Training', 'lambda_bal'))
        parser.add_argument('--lambda_balance', type=float,
                  default=config.getfloat('Training', 'lambda_balance', fallback=config.getfloat('Training', 'lambda_bal', fallback=0.1)))
        parser.add_argument('--lambda_pred', type=float,
                    default=config.getfloat('Training', 'lambda_pred', fallback=0.3))
        parser.add_argument('--lambda_amp', type=float,
                    default=config.getfloat('Training', 'lambda_amp', fallback=2.0))
        parser.add_argument('--lambda_scale', type=float,
                    default=config.getfloat('Training', 'lambda_scale', fallback=0.5))
        parser.add_argument('--lambda_msle', type=float,
                default=config.getfloat('Training', 'lambda_msle', fallback=5.0))
        parser.add_argument('--lambda_rank', type=float,
                default=config.getfloat('Training', 'lambda_rank', fallback=0.1))
        parser.add_argument('--lambda_tcl', type=float,
                default=config.getfloat('Training', 'lambda_tcl', fallback=0.1))
        parser.add_argument('--lambda_orth', type=float,
                default=config.getfloat('Training', 'lambda_orth', fallback=0.05))
        parser.add_argument('--contrastive_tau', type=float,
                default=config.getfloat('Training', 'contrastive_tau', fallback=0.2))
        parser.add_argument('--orth_eps', type=float,
                default=config.getfloat('Training', 'orth_eps', fallback=1e-6))
        parser.add_argument('--scale_epsilon', type=float,
                    default=config.getfloat('Training', 'scale_epsilon', fallback=1.0))
        parser.add_argument('--grad_clip', type=float, 
                          default=config.getfloat('Training', 'grad_clip'))
        parser.add_argument('--save_interval', type=int, 
                          default=config.getint('Training', 'save_interval'))
        parser.add_argument('--val_interval', type=int, 
                          default=config.getint('Training', 'val_interval'))
        parser.add_argument('--log_interval', type=int, 
                          default=config.getint('Training', 'log_interval'))
        parser.add_argument('--primary_metric', type=str,
                    default=config.get('Training', 'primary_metric', fallback='mape'))
        parser.add_argument('--composite_amp_weight', type=float,
                    default=config.getfloat('Training', 'composite_amp_weight', fallback=1.0))
        parser.add_argument('--mape_save_delta', type=float,
                    default=config.getfloat('Training', 'mape_save_delta', fallback=0.1))
        parser.add_argument('--warmup_epochs', type=int,
                    default=config.getint('Training', 'warmup_epochs', fallback=5))
        parser.add_argument('--cosine_tmax_multiplier', type=float,
                    default=config.getfloat('Training', 'cosine_tmax_multiplier', fallback=1.5))
    
    if 'System' in config:
        parser.add_argument('--device', type=str, default=config.get('System', 'device'))
        parser.add_argument('--num_workers', type=int, 
                          default=config.getint('System', 'num_workers'))
        parser.add_argument('--seed', type=int, default=config.getint('System', 'seed'))
    
    args, _ = parser.parse_known_args()
    return args


def print_config(args):
    """
    Print configuration in a readable format.
    """
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    print("="*60 + "\n")