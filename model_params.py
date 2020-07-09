SEQ2SEQ_MODEL_PARAMS = {
    'direction':'ua-eng',
    'num_examples': 20000,
    'test_size': 0.01,
    'embedding_dim': 256, 
    'units': 1024,
    'epochs': 5, 
    'batch_size': 64,
    'inp_tokenizer_file': 'input_tokenizer', 
    'targ_tokenizer_file': 'target_tokenizer',
    'model_checkpoint_dir': './seq2seq_checkpoints',
    'checkpoint_save_after_epochs': 5,
    'prediction_max_length_inp': 20,
    'prediction_max_length_targ': 20
}

TRANSFORMER_MODEL_PARAMS = {
    'direction':'ua-eng',
    'num_examples': 20000,
    'test_size': 0.2,
    'num_layers': 2,
    'd_model': 128, 
    'dff': 512, 
    'num_heads': 8, 
    'dropout_rate': 0.1,
    'epochs': 5, 
    'batch_size': 64,
    'optimizer_beta_1': 0.9, 
    'optimizer_beta_2': 0.98, 
    'optimizer_epsilon': 1e-9,
    'inp_tokenizer_file': 'input_tokenizer', 
    'targ_tokenizer_file': 'target_tokenizer',
    'model_checkpoint_dir': './transformer_checkpoints', 
    'checkpoint_save_after_epochs': 5, 
    'checkpoints_max_to_keep': 5,
    'prediction_max_length_inp': 20,
    'prediction_max_length_targ': 20
}
