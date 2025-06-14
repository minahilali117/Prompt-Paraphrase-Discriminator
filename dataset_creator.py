import json
import random
import pandas as pd
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
import itertools

class DatasetCreator:
    """Creates training and test datasets for prompt paraphrase discrimination."""
    
    def __init__(self):
        self.seed_prompts = self._load_seed_prompts()
        self.synonyms = self._load_synonyms()
        self.translations = self._load_translations()
        
    def _load_seed_prompts(self) -> Dict[str, List[str]]:
        """Load seed prompts categorized by scene type."""
        return {
            "objects": [
                "a red bicycle",
                "a blue car",
                "a green tree",
                "a yellow flower",
                "a white house",
                "a black cat",
                "a brown dog",
                "a silver laptop",
                "a wooden chair",
                "a glass window"
            ],
            "actions": [
                "running in the park",
                "sleeping on the couch",
                "flying in the sky",
                "swimming in the lake",
                "climbing the mountain",
                "reading a book",
                "playing music",
                "cooking dinner",
                "dancing happily",
                "walking slowly"
            ],
            "settings": [
                "under a tree",
                "on the beach",
                "in the forest",
                "near the river",
                "beside the road",
                "in the garden",
                "at the park",
                "on the mountain",
                "by the lake",
                "in the city"
            ],
            "weather": [
                "during sunset",
                "in the rain",
                "on a sunny day",
                "in the snow",
                "during storm",
                "at dawn",
                "in the fog",
                "under clear sky",
                "in moonlight",
                "during twilight"
            ]
        }
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """Load synonym mappings for paraphrase generation."""
        return {
            "red": ["crimson", "scarlet", "cherry", "ruby"],
            "blue": ["azure", "navy", "cerulean", "cobalt"],
            "green": ["emerald", "jade", "olive", "forest green"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["tiny", "little", "miniature", "petite"],
            "bicycle": ["bike", "cycle", "two-wheeler"],
            "car": ["vehicle", "automobile", "auto"],
            "dog": ["canine", "hound", "pup", "pooch"],
            "cat": ["feline", "kitten", "kitty"],
            "house": ["home", "residence", "dwelling"],
            "running": ["sprinting", "jogging", "racing"],
            "walking": ["strolling", "ambling", "pacing"],
            "under": ["beneath", "below", "underneath"],
            "on": ["upon", "atop", "over"],
            "near": ["close to", "beside", "next to"],
            "beautiful": ["gorgeous", "stunning", "lovely"],
            "old": ["ancient", "elderly", "aged"],
            "fast": ["quick", "rapid", "swift"]
        }
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translations for multilingual pairs."""
        return {
            "a red bicycle under a tree": {
                "es": "una bicicleta roja bajo un árbol",
                "fr": "un vélo rouge sous un arbre",
                "de": "ein rotes Fahrrad unter einem Baum",
                "it": "una bicicletta rossa sotto un albero",
                "pt": "uma bicicleta vermelha sob uma árvore"
            },
            "a blue car on the road": {
                "es": "un coche azul en la carretera",
                "fr": "une voiture bleue sur la route",
                "de": "ein blaues Auto auf der Straße",
                "it": "una macchina blu sulla strada",
                "pt": "um carro azul na estrada"
            },
            "a dog running in the park": {
                "es": "un perro corriendo en el parque",
                "fr": "un chien qui court dans le parc",
                "de": "ein Hund läuft im Park",
                "it": "un cane che corre nel parco",
                "pt": "um cachorro correndo no parque"
            },
            "a cat sleeping on the couch": {
                "es": "un gato durmiendo en el sofá",
                "fr": "un chat qui dort sur le canapé",
                "de": "eine Katze schläft auf dem Sofa",
                "it": "un gatto che dorme sul divano",
                "pt": "um gato dormindo no sofá"
            },
            "beautiful sunset over the ocean": {
                "es": "hermoso atardecer sobre el océano",
                "fr": "beau coucher de soleil sur l'océan",
                "de": "schöner Sonnenuntergang über dem Ozean",
                "it": "bellissimo tramonto sull'oceano",
                "pt": "belo pôr do sol sobre o oceano"
            }
        }
    
    def generate_scene_prompt(self) -> str:
        """Generate a random scene prompt by combining elements."""
        obj = random.choice(self.seed_prompts["objects"])
        action = random.choice(self.seed_prompts["actions"])
        setting = random.choice(self.seed_prompts["settings"])
        
        # Randomly decide if to include weather
        if random.random() < 0.3:
            weather = random.choice(self.seed_prompts["weather"])
            return f"{obj} {action} {setting} {weather}"
        else:
            return f"{obj} {action} {setting}"
    
    def create_paraphrase(self, prompt: str) -> str:
        """Create a paraphrase of the given prompt using synonyms."""
        words = prompt.split()
        paraphrased_words = []
        
        for word in words:
            # Remove common articles and prepositions from synonym replacement
            clean_word = word.lower().strip('.,!?')
            if clean_word in self.synonyms and random.random() < 0.7:
                paraphrased_words.append(random.choice(self.synonyms[clean_word]))
            else:
                paraphrased_words.append(word)
        
        return ' '.join(paraphrased_words)
    
    def create_structural_paraphrase(self, prompt: str) -> str:
        """Create structural paraphrases by reordering elements."""
        # Simple reordering patterns
        if "under" in prompt:
            parts = prompt.split("under")
            if len(parts) == 2:
                return f"beneath{parts[1]}, {parts[0].strip()}"
        
        if "on" in prompt:
            parts = prompt.split("on")
            if len(parts) == 2:
                return f"positioned on{parts[1]}, {parts[0].strip()}"
        
        # Fallback to synonym paraphrase
        return self.create_paraphrase(prompt)
    
    def create_positive_pairs(self, n_pairs: int) -> List[Tuple[str, str, int]]:
        """Create positive pairs (same scene descriptions)."""
        pairs = []
        
        # Generate base prompts
        base_prompts = [self.generate_scene_prompt() for _ in range(n_pairs // 3)]
        
        for prompt in base_prompts:
            # Type 1: Synonym paraphrases
            paraphrase1 = self.create_paraphrase(prompt)
            pairs.append((prompt, paraphrase1, 1))
            
            # Type 2: Structural paraphrases
            paraphrase2 = self.create_structural_paraphrase(prompt)
            pairs.append((prompt, paraphrase2, 1))
            
            # Type 3: Translation pairs (if available)
            if prompt in self.translations:
                lang = random.choice(list(self.translations[prompt].keys()))
                translation = self.translations[prompt][lang]
                pairs.append((prompt, translation, 1))
        
        # Add more translation pairs from predefined set
        for eng_prompt, translations in self.translations.items():
            for lang, translation in translations.items():
                if len(pairs) < n_pairs:
                    pairs.append((eng_prompt, translation, 1))
        
        return pairs[:n_pairs]
    
    def create_negative_pairs(self, n_pairs: int) -> List[Tuple[str, str, int]]:
        """Create negative pairs (different scene descriptions)."""
        pairs = []
        
        # Generate diverse prompts
        prompts = [self.generate_scene_prompt() for _ in range(n_pairs * 2)]
        
        # Create random pairings ensuring they're different
        random.shuffle(prompts)
        for i in range(0, len(prompts) - 1, 2):
            if i + 1 < len(prompts) and len(pairs) < n_pairs:
                prompt1, prompt2 = prompts[i], prompts[i + 1]
                # Ensure they're actually different
                if not self._are_scenes_similar(prompt1, prompt2):
                    pairs.append((prompt1, prompt2, 0))
        
        # Create more controlled negative examples
        base_prompts = list(self.translations.keys())
        for i in range(len(base_prompts)):
            for j in range(i + 1, len(base_prompts)):
                if len(pairs) < n_pairs:
                    pairs.append((base_prompts[i], base_prompts[j], 0))
        
        return pairs[:n_pairs]
    
    def _are_scenes_similar(self, prompt1: str, prompt2: str) -> bool:
        """Check if two prompts describe similar scenes (simple heuristic)."""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        # If they share more than 60% of words, consider them similar
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union > 0.6 if union > 0 else False
    
    def create_dataset(self, n_pairs: int, train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create complete dataset with train/test split."""
        # Create equal number of positive and negative pairs
        positive_pairs = self.create_positive_pairs(n_pairs // 2)
        negative_pairs = self.create_negative_pairs(n_pairs // 2)
        
        # Combine and shuffle
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Create DataFrame
        df = pd.DataFrame(all_pairs, columns=['prompt1', 'prompt2', 'label'])
        
        # Split into train and test
        split_idx = int(len(df) * train_ratio)
        train_df = df[:split_idx].reset_index(drop=True)
        test_df = df[split_idx:].reset_index(drop=True)
        
        return train_df, test_df
    
    def save_dataset(self, output_dir: str, n_pairs: int = 10000, train_ratio: float = 0.8):
        """Save dataset to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Creating dataset with {n_pairs} pairs...")
        train_df, test_df = self.create_dataset(n_pairs, train_ratio)
        
        # Save datasets
        train_path = output_path / "training_pairs.csv"
        test_path = output_path / "test_pairs.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Save seed data for reference
        seed_path = output_path / "seed_prompts.json"
        with open(seed_path, 'w') as f:
            json.dump({
                'seed_prompts': self.seed_prompts,
                'synonyms': self.synonyms,
                'translations': self.translations
            }, f, indent=2)
        
        print(f"Dataset saved to {output_dir}")
        print(f"Training pairs: {len(train_df)} ({train_df['label'].sum()} positive, {len(train_df) - train_df['label'].sum()} negative)")
        print(f"Test pairs: {len(test_df)} ({test_df['label'].sum()} positive, {len(test_df) - test_df['label'].sum()} negative)")
        
        return train_df, test_df

def main():
    parser = argparse.ArgumentParser(description='Create prompt paraphrase discrimination dataset')
    parser.add_argument('--output_dir', type=str, default='data/', 
                       help='Output directory for dataset files')
    parser.add_argument('--num_pairs', type=int, default=10000,
                       help='Total number of prompt pairs to generate')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create dataset
    creator = DatasetCreator()
    creator.save_dataset(args.output_dir, args.num_pairs, args.train_ratio)

if __name__ == "__main__":
    main()