import os
import csv
import re
import torch
import pickle
import json
import numpy as np
from typing import Dict, List, Union, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
import io
from contextlib import asynccontextmanager

# ===============================================================
# Class Definition for the CharacterVocabulary
# ===============================================================
class CharacterVocabulary:
    def __init__(self, alphabet=None):
        self.default_alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.char_to_idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx_to_char = {0: self.pad_token, 1: self.unk_token}
        self.n_chars = 2
        self.char_count = {}
        
        if alphabet is not None:
            self.use_fixed_alphabet(alphabet)

    def use_fixed_alphabet(self, alphabet=None):
        if alphabet is None:
            alphabet = self.default_alphabet
        
        print(f"Using fixed alphabet with {len(alphabet)} characters")
        
        for char in alphabet:
            if char not in self.char_to_idx:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.n_chars += 1
                
        print(f"Vocabulary initialized with {self.n_chars} characters (including special tokens)")

    def build_vocab(self, texts, min_count=1):
        print("Building character vocabulary from data...")
        
        for text in texts:
            for char in text:
                if char not in self.char_count:
                    self.char_count[char] = 0
                self.char_count[char] += 1
                
        for char, count in self.char_count.items():
            if count >= min_count:
                self.char_to_idx[char] = self.n_chars
                self.idx_to_char[self.n_chars] = char
                self.n_chars += 1
                
        print(f"Vocabulary built with {self.n_chars} characters")

    def encode_text(self, text, max_len=300):
        indices = np.full(max_len, self.char_to_idx[self.pad_token], dtype=np.int64)
        
        for i, char in enumerate(text[:max_len]):
            indices[i] = self.char_to_idx.get(char, self.char_to_idx[self.unk_token])
            
        return indices

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

# ===============================================================
# Model Architecture (Recreating EXACT original architecture)
# ===============================================================

# Custom CNN Layer class
class CNNLayer(torch.nn.Module):
    def __init__(self, input_channels, large_features, small_features,
                 kernel_size, pool_size=None, batch_norm=False):
        super(CNNLayer, self).__init__()
        
        # Store channels for residual connection checking
        self.input_channels = input_channels
        self.output_channels = small_features
        
        # Primary convolution
        self.conv = torch.nn.Conv1d(
            in_channels=input_channels,
            out_channels=large_features,
            kernel_size=kernel_size,
            padding=kernel_size // 2  # Same padding
        )
        
        # Batch normalization (optional)
        self.batch_norm = torch.nn.BatchNorm1d(large_features) if batch_norm else None
        
        # Pooling layer (optional)
        self.pool = torch.nn.MaxPool1d(kernel_size=pool_size, stride=pool_size) if pool_size is not None else None
        
        # Dimension reduction layer
        self.reduce = torch.nn.Conv1d(
            in_channels=large_features,
            out_channels=small_features,
            kernel_size=1  # 1x1 convolution
        )
        
        # Batch normalization for reduction (optional)
        self.reduce_bn = torch.nn.BatchNorm1d(small_features) if batch_norm else None

    def forward(self, x):
        # Store input for potential residual connection
        residual = x if self.input_channels == self.output_channels else None
        
        # Apply convolution
        x = self.conv(x)
        
        # Apply batch normalization if present
        if self.batch_norm is not None:
            x = self.batch_norm(x)
            
        # Apply ReLU activation
        x = torch.nn.functional.relu(x)
        
        # Apply pooling if it exists
        if self.pool is not None:
            x = self.pool(x)
            # Cannot use residual connection if shape changed by pooling
            residual = None
            
        # Apply dimension reduction
        x_reduced = self.reduce(x)
        
        # Apply batch normalization for reduction if present
        if self.reduce_bn is not None:
            x_reduced = self.reduce_bn(x_reduced)
            
        # Apply ReLU to reduced features
        x_reduced = torch.nn.functional.relu(x_reduced)
        
        # Add residual connection if possible (shapes must match)
        if residual is not None:
            x = x_reduced + residual
        else:
            x = x_reduced
            
        return x

# Custom FC Layer class
class FCLayer(torch.nn.Module):
    def __init__(self, input_units, large_units, small_units, batch_norm=False, dropout_rate=0.3):
        super(FCLayer, self).__init__()
        
        # Large unit expansion
        self.large = torch.nn.Conv1d(
            in_channels=input_units,
            out_channels=large_units,
            kernel_size=1
        )
        
        # Batch normalization (optional)
        self.large_bn = torch.nn.BatchNorm1d(large_units) if batch_norm else None
        
        # Small unit reduction
        self.small = torch.nn.Conv1d(
            in_channels=large_units,
            out_channels=small_units,
            kernel_size=1
        )
        
        # Batch normalization (optional)
        self.small_bn = torch.nn.BatchNorm1d(small_units) if batch_norm else None
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        # Apply large unit expansion
        x_large = self.large(x)
        
        # Apply batch normalization if present
        if self.large_bn is not None:
            x_large = self.large_bn(x_large)
            
        x_large = torch.nn.functional.relu(x_large)
        x_large = self.dropout(x_large)
        
        # Apply small unit reduction
        x_small = self.small(x_large)
        
        # Apply batch normalization if present
        if self.small_bn is not None:
            x_small = self.small_bn(x_small)
            
        x = torch.nn.functional.relu(x_small)
        
        return x

class AttentionLayer(torch.nn.Module):
    """Multi-level attention layer for focusing on specific patterns in sequence data."""
    def __init__(self, hidden_dim, attention_dim=64, name=None):
        super(AttentionLayer, self).__init__()
        self.name = name  # For tracking which attention layer is active
        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.attention_projection = torch.nn.Linear(hidden_dim, self.attention_dim)
        self.attention_context = torch.nn.Linear(self.attention_dim, 1, bias=False)
        
        # Initialize weights
        torch.nn.init.xavier_normal_(self.attention_projection.weight)
        torch.nn.init.xavier_normal_(self.attention_context.weight)

    def forward(self, sequence):
        # sequence shape: [batch_size, seq_len, hidden_dim]
        
        # Project hidden states
        # [batch_size, seq_len, attention_dim]
        projection = torch.tanh(self.attention_projection(sequence))
        
        # Get attention weights
        # [batch_size, seq_len, 1]
        weights = self.attention_context(projection)
        
        # Apply softmax to get attention distribution
        # [batch_size, seq_len, 1]
        weights = torch.nn.functional.softmax(weights, dim=1)
        
        # Apply attention weights
        # [batch_size, seq_len, hidden_dim] * [batch_size, seq_len, 1] -> [batch_size, seq_len, hidden_dim]
        weighted_hidden = sequence * weights
        
        # Sum over sequence dimension to get context vector
        # [batch_size, hidden_dim]
        context_vector = torch.sum(weighted_hidden, dim=1)
        
        # Return both the context vector and attention weights for visualization
        return context_vector, weights

class EnhancedCharCNNBiLSTM(torch.nn.Module):
    def __init__(self, n_chars, n_classes, config):
        super(EnhancedCharCNNBiLSTM, self).__init__()
        
        # Get configuration parameters
        char_emb_dim = config.get('char_emb_dim', 50)
        lstm_hidden_dim = config.get('lstm_hidden_dim', 128)
        dropout_rate = config.get('dropout_rate', 0.3)
        cnn_layers_config = config.get('char_cnn_layers', [])
        fc_layers_config = config.get('fc_layers', [])
        
        # Character embedding layer
        self.char_embedding = torch.nn.Embedding(n_chars, char_emb_dim, padding_idx=0)
        
        # Build the CNN layers using custom CNNLayer class
        self.cnn_layers = torch.nn.ModuleList()
        input_channels = char_emb_dim
        
        # Create each CNN layer based on the architecture
        for layer_config in cnn_layers_config:
            # Create a proper PyTorch Module for the CNN layer
            cnn_layer = CNNLayer(
                input_channels=input_channels,
                large_features=layer_config['large_features'],
                small_features=layer_config['small_features'],
                kernel_size=layer_config['kernel'],
                pool_size=layer_config.get('pool'),
                batch_norm=layer_config.get('batch_norm', False)
            )
            
            # Add to module list
            self.cnn_layers.append(cnn_layer)
            
            # Update input channels for next layer
            input_channels = layer_config['small_features']
            
        # FC Layers using custom FCLayer class
        self.fc_layers = torch.nn.ModuleList()
        for i, layer_config in enumerate(fc_layers_config):
            # Calculate input units for this layer
            current_in_channels = input_channels if i == 0 else fc_layers_config[i-1]['small_units']
            
            # Create FC layer
            fc_layer = FCLayer(
                input_units=current_in_channels,
                large_units=layer_config['large_units'],
                small_units=layer_config['small_units'],
                batch_norm=layer_config.get('batch_norm', False),
                dropout_rate=dropout_rate
            )
            
            # Add to module list
            self.fc_layers.append(fc_layer)
            
        # The last FC layer's output channels become the input to the LSTM
        final_fc_output = fc_layers_config[-1]['small_units'] if fc_layers_config else input_channels
        
        # BiLSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=final_fc_output,
            hidden_size=lstm_hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout_rate)
        
        # ATTENTION LAYERS
        use_attention = config.get('use_attention', True)
        attention_dim = config.get('attention_dim', 64)
        
        if use_attention:
            # 3 attention layers for toxicity levels
            self.attention_not_toxic = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="not_toxic")
            self.attention_toxic = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="toxic")
            self.attention_very_toxic = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="very_toxic")
            
            # 4 attention layers for categories
            self.attention_insult = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="insult")
            self.attention_profanity = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="profanity")
            self.attention_threat = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="threat")
            self.attention_identity_hate = AttentionLayer(lstm_hidden_dim * 2, attention_dim, name="identity_hate")
            
            # Specialized category outputs with attention
            self.fc_insult = torch.nn.Linear(lstm_hidden_dim * 2, 1)
            self.fc_profanity = torch.nn.Linear(lstm_hidden_dim * 2, 1)
            self.fc_threat = torch.nn.Linear(lstm_hidden_dim * 2, 1)
            self.fc_identity_hate = torch.nn.Linear(lstm_hidden_dim * 2, 1)
            
        # Output layers
        self.fc_toxicity = torch.nn.Linear(lstm_hidden_dim * 2, 3)  # 3 toxicity levels
        self.fc_category = torch.nn.Linear(lstm_hidden_dim * 2, 4)  # 4 categories
        
        # For tracking attention distributions
        self.attention_weights = {}

    def forward(self, char_ids):
        # Character embeddings (batch_size, seq_len, char_emb_dim)
        char_embeds = self.char_embedding(char_ids)
        
        # Convolutional layers expect (batch_size, channel, seq_len)
        # So we transpose the embedding dimensions
        x = char_embeds.permute(0, 2, 1)
        
        # Apply each CNN layer
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            
        # Apply each FC layer (as 1x1 convolutions)
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
            
        # Permute back for LSTM: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # Apply BiLSTM
        lstm_out, _ = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Global max pooling over sequence dimension (for backward compatibility)
        global_max_pool, _ = torch.max(lstm_out, dim=1)
        
        # Store attention outputs and weights
        attention_outputs = {}
        self.attention_weights = {}
        
        # Apply toxicity-level attention mechanisms
        attention_outputs['not_toxic'], self.attention_weights['not_toxic'] = self.attention_not_toxic(lstm_out)
        attention_outputs['toxic'], self.attention_weights['toxic'] = self.attention_toxic(lstm_out)
        attention_outputs['very_toxic'], self.attention_weights['very_toxic'] = self.attention_very_toxic(lstm_out)
        
        # Apply category-specific attention mechanisms
        attention_outputs['insult'], self.attention_weights['insult'] = self.attention_insult(lstm_out)
        attention_outputs['profanity'], self.attention_weights['profanity'] = self.attention_profanity(lstm_out)
        attention_outputs['threat'], self.attention_weights['threat'] = self.attention_threat(lstm_out)
        attention_outputs['identity_hate'], self.attention_weights['identity_hate'] = self.attention_identity_hate(lstm_out)
        
        # For toxicity classification, blend global pooling with attention outputs
        toxicity_features = global_max_pool * 0.5 + (
            attention_outputs['not_toxic'] * 0.2 +
            attention_outputs['toxic'] * 0.2 +
            attention_outputs['very_toxic'] * 0.1
        )
        
        # Calculate category outputs using dedicated attention heads
        insult_output = self.fc_insult(attention_outputs['insult'])
        profanity_output = self.fc_profanity(attention_outputs['profanity'])
        threat_output = self.fc_threat(attention_outputs['threat'])
        identity_hate_output = self.fc_identity_hate(attention_outputs['identity_hate'])
        
        # Concatenate category outputs
        category_output = torch.cat([
            insult_output, profanity_output, threat_output, identity_hate_output
        ], dim=1)
        
        # Apply toxicity layer to attention-enhanced features
        toxicity_output = self.fc_toxicity(toxicity_features)
        
        return toxicity_output, category_output

# Global variables to store model and related components
MODEL = None
CHAR_VOCAB = None
CONFIG = {}
TEXT_CENSOR = None
DEVICE = None

# ===============================================================
# Model Loading and Prediction Utilities
# ===============================================================

def load_model_components(model_dir: str):
    """
    Load the saved toxicity detection model, vocabulary, and configuration.
    """
    global MODEL, CHAR_VOCAB, CONFIG, DEVICE
    
    print(f"Loading model from {model_dir}...")
    
    # Set device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load model config
    try:
        with open(os.path.join(model_dir, 'model_config.json'), 'r') as f:
            model_config = json.load(f)
            print("Loaded model configuration")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model configuration file not found in {model_dir}")
    
    # Load training config if available
    try:
        with open(os.path.join(model_dir, 'training_config.json'), 'r') as f:
            training_config = json.load(f)
            print("Loaded training configuration")
    except FileNotFoundError:
        training_config = {}
        print("No training configuration found, using defaults")
    
    # Merge configurations
    CONFIG.update(model_config)
    CONFIG.update(training_config)
    
    # Load character vocabulary
    try:
        vocab_path = os.path.join(model_dir, 'char_vocab.pkl')
        with open(vocab_path, 'rb') as f:
            pickle_data = f.read()
        
        # Custom unpickler that handles class lookup
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'CharacterVocabulary':
                    return CharacterVocabulary
                return super().find_class(module, name)
        
        # Use custom unpickler to load vocabulary
        CHAR_VOCAB = CustomUnpickler(io.BytesIO(pickle_data)).load()
        print("Loaded character vocabulary")
    except Exception as e:
        print(f"Error loading character vocabulary: {e}")
        raise
    
    # Initialize model with the EXACT same architecture
    try:
        MODEL = EnhancedCharCNNBiLSTM(
            n_chars=CHAR_VOCAB.n_chars,
            n_classes=5,  # 1 toxicity level (3 classes) + 4 binary categories
            config=CONFIG
        ).to(DEVICE)
        
        # Load model weights
        model_weights_path = os.path.join(model_dir, 'model.pth')
        MODEL.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
        MODEL.eval()  # Set to evaluation mode
        print("Loaded model weights successfully")
    except Exception as e:
        print(f"Detailed error loading model: {str(e)}")
        raise Exception(f"Error loading model: {str(e)}")
    
    print("Model and components loaded successfully!")

def predict_text(text: str) -> Dict[str, Any]:
    """
    Predict toxicity for a single text without including percentages.
    """
    global MODEL, CHAR_VOCAB, DEVICE, CONFIG
    
    if MODEL is None or CHAR_VOCAB is None:
        raise ValueError("Model or vocabulary not loaded")
    
    # Preprocess text
    text = preprocess_text(text)
    
    # Encode text to character indices
    char_ids = CHAR_VOCAB.encode_text(text, CONFIG.get('max_chars', 300))
    char_ids_tensor = torch.tensor([char_ids], dtype=torch.long).to(DEVICE)
    
    # Set model to evaluation mode
    MODEL.eval()
    
    # Make prediction
    with torch.no_grad():
        toxicity_output, category_output = MODEL(char_ids_tensor)
        
        # Apply temperature scaling if configured
        temp = CONFIG.get('temperature_scaling', 1.0)
        if temp != 1.0:
            toxicity_output = toxicity_output / temp
            category_output = category_output / temp
        
        # Get toxicity prediction
        toxicity_probs = torch.softmax(toxicity_output, dim=1)
        toxicity_pred = torch.argmax(toxicity_probs, dim=1).item()
        
        # Get category predictions
        category_probs = torch.sigmoid(category_output)
        category_thresholds = CONFIG.get('category_thresholds', [0.6, 0.6, 0.55, 0.7])
        category_preds = []
        
        for i, threshold in enumerate(category_thresholds):
            if i < category_probs.size(1):
                category_preds.append((category_probs[0, i] > threshold).item())
    
    # Convert predictions to human-readable format
    toxicity_levels = ['not toxic', 'toxic', 'very toxic']
    category_labels = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
    
    # Create simplified result without percentages
    result = {
        'toxicity': toxicity_levels[toxicity_pred],
        'categories': [
            category_labels[i] for i in range(len(category_preds)) if category_preds[i]
        ]
    }
    
    return result

def batch_predict_texts(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Predict toxicity for multiple texts without including percentages.
    """
    global MODEL, CHAR_VOCAB, DEVICE, CONFIG
    
    if MODEL is None or CHAR_VOCAB is None:
        raise ValueError("Model or vocabulary not loaded")
    
    results = []
    batch_size = 32  # Process in batches to avoid memory issues
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_size_actual = len(batch_texts)
        
        # Preprocess texts
        preprocessed_texts = [preprocess_text(text) for text in batch_texts]
        
        # Encode texts
        char_ids_list = [
            CHAR_VOCAB.encode_text(text, CONFIG.get('max_chars', 300))
            for text in preprocessed_texts
        ]
        char_ids_tensor = torch.tensor(char_ids_list, dtype=torch.long).to(DEVICE)
        
        # Make predictions
        MODEL.eval()
        with torch.no_grad():
            toxicity_outputs, category_outputs = MODEL(char_ids_tensor)
            
            # Apply temperature scaling if configured
            temp = CONFIG.get('temperature_scaling', 1.0)
            if temp != 1.0:
                toxicity_outputs = toxicity_outputs / temp
                category_outputs = category_outputs / temp
            
            # Get toxicity predictions
            toxicity_probs = torch.softmax(toxicity_outputs, dim=1)
            toxicity_preds = torch.argmax(toxicity_probs, dim=1).cpu().numpy()
            
            # Get category predictions
            category_probs = torch.sigmoid(category_outputs)
            category_thresholds = CONFIG.get('category_thresholds', [0.6, 0.6, 0.55, 0.7])
            
            for j in range(batch_size_actual):
                # Get toxicity prediction
                toxicity_pred = toxicity_preds[j]
                
                # Get category predictions
                category_preds = []
                for k, threshold in enumerate(category_thresholds):
                    if k < category_probs.size(1):
                        category_preds.append((category_probs[j, k] > threshold).item())
                
                # Convert predictions to human-readable format
                toxicity_levels = ['not toxic', 'toxic', 'very toxic']
                category_labels = CONFIG.get('category_columns', ['insult', 'profanity', 'threat', 'identity_hate'])
                
                # Create simplified result without percentages
                result = {
                    'toxicity': toxicity_levels[toxicity_pred],
                    'categories': [
                        category_labels[k] for k in range(len(category_preds)) if category_preds[k]
                    ]
                }
                
                results.append(result)
    
    return results

# Utility function for text preprocessing
def preprocess_text(text):
    """Preprocess text before prediction."""
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ===============================================================
# Censoring Utilities
# ===============================================================

class TextCensor:
    """Class to handle text censoring based on model predictions and word lists."""
    
    def __init__(self, censor_words_path: str = None, additional_censor_path: str = None):
        """Initialize the censor with optional path to words list CSV."""
        self.censor_words = set()
        self.additional_censor_words = set()  # New set for additional words
        self.censor_patterns = []
        self.additional_censor_patterns = []  # New patterns list
        
        if censor_words_path:
            self.load_censor_words(censor_words_path)
        else:
            print("No censor words path provided. Creating empty censor.")
            
        if additional_censor_path:
            self.load_additional_censor_words(additional_censor_path)
        else:
            print("No additional censor words path provided.")
    
    def load_censor_words(self, path: str):
        """Load words to censor from a CSV file."""
        try:
            print(f"Attempting to load censor words from: {path}")
            
            if not os.path.exists(path):
                print(f"WARNING: Censor words file not found at {path}")
                # Create a minimal default set of words to censor
                self.censor_words = {"fuck", "shit", "ass", "damn", "bitch", "cunt", "dick"}
                print(f"Using {len(self.censor_words)} default censor words")
            else:
                # Try to read the file
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"Successfully read file with {len(content)} characters")
                    
                    # If file is empty or too small, use defaults
                    if len(content.strip()) < 10:
                        print("File content too small, using defaults")
                        self.censor_words = {"fuck", "shit", "ass", "damn", "bitch", "cunt", "dick"}
                    else:
                        # Try to parse as CSV
                        words = []
                        try:
                            # Try to use csv module
                            f.seek(0)  # Reset file pointer
                            reader = csv.reader(f)
                            for row in reader:
                                if row and row[0].strip():
                                    # Skip header row if it exists
                                    if row[0].lower() == "word" and reader.line_num == 1:
                                        continue
                                    words.append(row[0].strip().lower())
                        except Exception as csv_err:
                            print(f"CSV parsing failed: {csv_err}")
                            # Fallback to simple line-by-line reading
                            f.seek(0)  # Reset file pointer
                            for line in f:
                                line = line.strip()
                                if line and line.lower() != "word":  # Skip header
                                    words.append(line.lower())
                        
                        # If we got words, use them
                        if words:
                            self.censor_words = set(words)
                            print(f"Loaded {len(self.censor_words)} words to censor")
                        else:
                            # Fallback to defaults
                            print("No words loaded from file, using defaults")
                            self.censor_words = {"fuck", "shit", "ass", "damn", "bitch", "cunt", "dick"}
            
            # Compile regex patterns for each word with word boundaries
            self.censor_patterns = []
            for word in self.censor_words:
                if word:  # Skip empty strings
                    try:
                        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                        self.censor_patterns.append(pattern)
                    except re.error:
                        print(f"Invalid regex pattern for word: {word}")
            
            print(f"Compiled {len(self.censor_patterns)} regex patterns")
                
        except Exception as e:
            print(f"Error loading censor words: {str(e)}")
            # Fallback to defaults
            self.censor_words = {"fuck", "shit", "ass", "damn", "bitch", "cunt", "dick"}
            self.censor_patterns = [
                re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE) 
                for word in self.censor_words
            ]
            print(f"Using {len(self.censor_words)} default censor words after error")
    
    def load_additional_censor_words(self, path: str):
        """Load additional words to censor from a CSV file."""
        try:
            print(f"Attempting to load additional censor words from: {path}")
            
            if not os.path.exists(path):
                print(f"WARNING: Additional censor words file not found at {path}")
                return
            
            # Try to read the file
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"Successfully read file with {len(content)} characters")
                
                # If file is empty or too small, return
                if len(content.strip()) < 10:
                    print("Additional file content too small, skipping")
                    return
                
                # Try to parse as CSV
                words = []
                try:
                    # Try to use csv module
                    f.seek(0)  # Reset file pointer
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0].strip():
                            # Skip header row if it exists
                            if row[0].lower() == "word" and reader.line_num == 1:
                                continue
                            words.append(row[0].strip().lower())
                except Exception as csv_err:
                    print(f"CSV parsing failed: {csv_err}")
                    # Fallback to simple line-by-line reading
                    f.seek(0)  # Reset file pointer
                    for line in f:
                        line = line.strip()
                        if line and line.lower() != "word":  # Skip header
                            words.append(line.lower())
                
                # If we got words, use them
                if words:
                    self.additional_censor_words = set(words)
                    print(f"Loaded {len(self.additional_censor_words)} additional words to censor")
                
                # Compile regex patterns for each word with word boundaries
                self.additional_censor_patterns = []
                for word in self.additional_censor_words:
                    if word:  # Skip empty strings
                        try:
                            pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                            self.additional_censor_patterns.append(pattern)
                        except re.error:
                            print(f"Invalid regex pattern for word: {word}")
                
                print(f"Compiled {len(self.additional_censor_patterns)} additional regex patterns")
                    
        except Exception as e:
            print(f"Error loading additional censor words: {str(e)}")
    
    def censor_text(self, text: str, prediction: Dict[str, Any] = None) -> str:
        """
        Censor text based on wordlist and optional prediction results.
        
        Args:
            text: The text to censor
            prediction: Optional prediction results to guide censoring
            
        Returns:
            Censored text
        """
        if not text:
            return text
        
        print(f"Censoring text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Make a copy of the original text
        censored_text = text
        original_text = text  # Save a copy of the original text
        
        # Track if any censoring was applied
        censoring_applied = False
        
        # First, censor based on word list using regex
        for pattern in self.censor_patterns:
            # Count occurrences before censoring
            count_before = len(re.findall(pattern, censored_text))
            
            if count_before > 0:
                censored_text = pattern.sub(lambda m: '*' * len(m.group(0)), censored_text)
                censoring_applied = True
        
        # If prediction is provided, apply additional censoring based on toxicity level
        model_indicates_toxicity = False
        if prediction:
            toxicity = prediction.get('toxicity', '')
            categories = prediction.get('categories', [])
            
            # Check if model indicates toxicity
            if toxicity in ['toxic', 'very toxic'] or any(cat in ['profanity', 'insult', 'threat', 'identity_hate'] for cat in categories):
                model_indicates_toxicity = True
            
            # If text is very toxic or contains certain categories, apply stronger censoring
            if toxicity == 'very toxic' or any(cat in ['threat', 'identity_hate'] for cat in categories):
                # Apply more aggressive censoring for very toxic content
                words = re.findall(r'\b\w+\b', censored_text)
                
                # Censor words that might be offensive (longer words that aren't already censored)
                for word in words:
                    if len(word) > 3 and '*' not in word:
                        # Check if this word appears in a toxic context
                        if toxicity == 'very toxic' and len(word) > 4:
                            pattern = re.compile(r'\b' + re.escape(word) + r'\b')
                            original = censored_text
                            censored_text = pattern.sub('*' * len(word), censored_text)
                            if original != censored_text:
                                censoring_applied = True
        
        # If model indicates toxicity but no censoring was applied, 
        # check the additional censor list as a fallback
        if model_indicates_toxicity and not censoring_applied and self.additional_censor_patterns:
            print("Model indicates toxicity but no censoring applied. Checking additional censor list.")
            
            # Check against the updated_censor.csv list
            for pattern in self.additional_censor_patterns:
                # Count occurrences
                count_before = len(re.findall(pattern, censored_text))
                
                if count_before > 0:
                    censored_text = pattern.sub(lambda m: '*' * len(m.group(0)), censored_text)
                    censoring_applied = True
        
        # Log results
        if censoring_applied:
            print(f"Censoring applied. Result: {censored_text[:50]}{'...' if len(censored_text) > 50 else ''}")
        else:
            print("No censoring needed for this text")
            
        return censored_text
# ===============================================================
# FastAPI Application
# ===============================================================

# Models for request/response
class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    toxicity: str
    categories: List[str]
    censored_text: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Declare global variables at the start of the function
    global MODEL, CHAR_VOCAB, CONFIG, TEXT_CENSOR
    
    # Startup: load model and censoring utilities
    model_dir = os.environ.get("MODEL_DIR", "output_model")
    
    # First look for censor_words.csv in the current directory
    if os.path.exists("censor_words.csv"):
        censor_words_path = "censor_words.csv"
    else:
        # Then try the model directory
        censor_words_path = os.path.join(model_dir, "extended_profanity_list.csv")
        if not os.path.exists(censor_words_path):
            # Check other common locations
            censor_words_path = os.path.join("output_model", "output_model/extended_profanity_list.csv")
    
    # Check for updated_censor.csv file
    additional_censor_path = "updated_censor.csv" if os.path.exists("updated_censor.csv") else None
    
    print(f"Loading model from directory: {model_dir}")
    print(f"Loading censor words from: {censor_words_path}")
    if additional_censor_path:
        print(f"Loading additional censor words from: {additional_censor_path}")
    else:
        print("No additional censor words file found. Create 'updated_censor.csv' to add words.")
    
    # Load model components
    load_model_components(model_dir)
    
    # Initialize text censor with both files
    TEXT_CENSOR = TextCensor(censor_words_path, additional_censor_path)
    
    # Print confirmation of initialization
    print(f"TEXT_CENSOR initialized with {len(TEXT_CENSOR.censor_words)} primary words and {len(TEXT_CENSOR.additional_censor_words) if hasattr(TEXT_CENSOR, 'additional_censor_words') else 0} additional words")
    
    yield
    
    # Shutdown: cleanup resources
    MODEL = None
    CHAR_VOCAB = None
    CONFIG = {}
    TEXT_CENSOR = None

# Create FastAPI application
app = FastAPI(
    title="Toxicity Detection API",
    description="API for detecting and censoring toxic content",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route that redirects to Swagger UI docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# Routes
# Update the predict endpoint to include censoring
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    Predict toxicity level and categories for a single text,
    and return a censored version of the text.
    """
    try:
        # Make prediction
        prediction = predict_text(request.text)
        
        # Apply censoring
        censored_text = TEXT_CENSOR.censor_text(request.text, prediction)
        
        # Return result with censoring
        return PredictionResponse(
            toxicity=prediction['toxicity'],
            categories=prediction['categories'],
            censored_text=censored_text
        )
    except Exception as e:
        print(f"Error in predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Update the batch predict endpoint to include censoring as well
@app.post("/predict-batch", response_model=List[PredictionResponse])
async def predict_batch(request: BatchTextRequest):
    """
    Predict toxicity for multiple texts and return censored versions.
    """
    try:
        # Make batch predictions
        predictions = batch_predict_texts(request.texts)
        
        # Create result list with censoring
        results = []
        for i, pred in enumerate(predictions):
            # Get the original text
            original_text = request.texts[i]
            
            # Apply censoring
            censored_text = TEXT_CENSOR.censor_text(original_text, pred)
            
            # Add to results
            results.append(
                PredictionResponse(
                    toxicity=pred['toxicity'],
                    categories=pred['categories'],
                    censored_text=censored_text
                )
            )
        
        return results
    except Exception as e:
        print(f"Error in batch predict endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# Keep the censor endpoint for backward compatibility
@app.post("/censor", response_model=PredictionResponse)
async def censor_text_endpoint(request: TextRequest):
    """
    Predict toxicity and censor a single text.
    This endpoint is maintained for backward compatibility.
    """
    return await predict(request)

@app.post("/upload-censor-words")
async def upload_censor_words(file: UploadFile = File(...)):
    """
    Upload a new CSV file of words to censor.
    """
    try:
        # Save uploaded file
        file_path = "uploaded_censor_words.csv"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Reload censor words
        global TEXT_CENSOR
        TEXT_CENSOR.load_censor_words(file_path)
        
        return {"message": f"Successfully loaded {len(TEXT_CENSOR.censor_words)} censor words"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading censor words: {str(e)}")

@app.post("/upload-additional-censor-words")
async def upload_additional_censor_words(file: UploadFile = File(...)):
    """
    Upload a new CSV file of additional words to censor.
    These words will be used when the model predicts toxicity but doesn't censor anything.
    """
    try:
        # Save uploaded file
        file_path = "updated_censor.csv"
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Reload additional censor words
        global TEXT_CENSOR
        TEXT_CENSOR.load_additional_censor_words(file_path)
        
        return {"message": f"Successfully loaded {len(TEXT_CENSOR.additional_censor_words)} additional censor words"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading additional censor words: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify the API is running and model is loaded.
    """
    if MODEL is None or CHAR_VOCAB is None:
        return {"status": "error", "message": "Model not loaded properly"}
    return {"status": "ok", "message": "API is running and model is loaded"}

# Run the application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("final_app:app", host="127.0.0.1", port=port, reload=True)