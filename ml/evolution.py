import json
import random
import os
import logging
import pandas as pd
import numpy as np
from ml.knowledge import KnowledgeBase

# Konfiguracja loggera
logger = logging.getLogger("EVOLUTION")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class GeneticEvolution:
    """
    THE BIOLAB V1.
    System Ewolucyjny do optymalizacji parametrów bota w oparciu o dobór naturalny.
    """
    GENOME_FILE = "assets/genome_v1.json"
    
    # Bazowy genom (Adam)
    DEFAULT_GENOME = {
        "min_confidence": 0.60,
        "risk_per_trade": 0.02,
        "sl_atr_mult": 1.5,
        "tp_atr_mult": 2.5,
        "whale_trust_factor": 0.5,
        "max_open_positions": 3
    }

    # Limity ewolucyjne (Hard Limits) to prevent catastrophic mutations
    LIMITS = {
        "min_confidence": (0.40, 0.95),
        "risk_per_trade": (0.005, 0.10), # Max 10% risk
        "sl_atr_mult": (0.5, 4.0),
        "tp_atr_mult": (1.0, 6.0),
        "whale_trust_factor": (0.0, 1.0),
        "max_open_positions": (1, 10)
    }

    @staticmethod
    def load_genome():
        """Ładuje aktualny 'najlepszy' genom lub zwraca domyślny."""
        if os.path.exists(GeneticEvolution.GENOME_FILE):
            try:
                with open(GeneticEvolution.GENOME_FILE, "r") as f:
                    genome = json.load(f)
                    # Merge with default to ensure all keys exist (migration safety)
                    full_genome = GeneticEvolution.DEFAULT_GENOME.copy()
                    full_genome.update(genome)
                    return full_genome
            except Exception as e:
                logger.error(f"Failed to load genome: {e}")
        return GeneticEvolution.DEFAULT_GENOME.copy()

    @staticmethod
    def mutate(genome):
        """
        Tworzy zmutowany wariant konfiguracji.
        Strategia: 'Point Mutation' - zmiana losowych genów o +/- 5-15%.
        """
        mutant = genome.copy()
        
        # Wybieramy liczbę genów do mutacji (1 do 3)
        genes_to_mutate = random.sample(list(GeneticEvolution.LIMITS.keys()), k=random.randint(1, 3))
        
        for gene in genes_to_mutate:
            current_val = mutant[gene]
            # Mutacja o +/- 5% do 15%
            mutation_factor = random.uniform(0.85, 1.15)
            new_val = current_val * mutation_factor
            
            # Application of Hard Limits
            min_val, max_val = GeneticEvolution.LIMITS[gene]
            
            # Special handling for integers
            if isinstance(GeneticEvolution.DEFAULT_GENOME[gene], int):
                new_val = int(round(new_val))
                new_val = max(int(min_val), min(int(max_val), new_val))
            else:
                new_val = max(min_val, min(max_val, new_val))
                new_val = round(new_val, 4) # Clean float
            
            mutant[gene] = new_val
            
        return mutant

    @staticmethod
    def calculate_fitness(genome, history_df):
        """
        Symuluje wynik finansowy danego genomu na danych historycznych.
        
        Arg:
            genome: Dict z parametrami
            history_df: DataFrame z KnowledgeBase (wymaga kolumn 'conf', 'target')
            
        Returns:
            Score (float): Całkowity wynik punktowy (np. suma R-multiple)
        """
        if history_df.empty:
            return 0.0

        score = 0.0
        trades_count = 0
        
        # Parametry genomu
        min_conf = genome["min_confidence"]
        # Uznajemy, że risk_per_trade jest stały dla score, 
        # ale liczymy 'R' (jednostki ryzyka)
        
        # Symulowany Reward/Risk Ratio
        # Zakładamy, że TP/SL są realizowane zgodnie z mnożnikami ATR
        # Średni Win = Risk * (TP/SL)
        # Średni Loss = Risk
        rr_ratio = genome["tp_atr_mult"] / genome["sl_atr_mult"]
        
        # Iteracja po historycznych setupach
        # W history_df mamy 'conf' (pewność sieci w momencie wejścia) i 'target' (czy trade był sukcesem 1.0 czy nie 0.0)
        # UWAGA: 'target' 1.0 = Win, 0.0 = Loss
        
        # Filtrujemy tylko te sygnały, w które ten konkretny mutant by wszedł
        # Zakładamy, że kolumna w bazie to 'conf' lub 'confidence'
        conf_col = 'conf' if 'conf' in history_df.columns else 'confidence'
        
        if conf_col not in history_df.columns or 'target' not in history_df.columns:
            # Fallback dla braku danych
            return -1.0

        for _, row in history_df.iterrows():
            signal_conf = row[conf_col]
            is_win = row['target'] == 1.0
            
            # Czy mutant wszedłby w ten trade?
            if signal_conf >= min_conf:
                trades_count += 1
                if is_win:
                    # Zysk: +1R * RR_Ratio
                    score += 1.0 * rr_ratio
                else:
                    # Strata: -1R
                    score -= 1.0
                    
        # Penalty za brak aktywności (overfitting do null)
        if trades_count < 5:
            score -= 10.0 # Kara za bycie "wyssuszoną" strategią
            
        return score

    @staticmethod
    def run_evolution_cycle(user_id="brain_v6"):
        """
        Główna pętla ewolucyjna.
        Tworzy populację mutantów, walczy na arenie historycznych danych, wybiera lidera.
        """
        logger.info("🧬 Starting Genetic Evolution Cycle...")
        
        # 1. Load Training Data (The Arena)
        try:
            df = KnowledgeBase.load_training_data(user_id)
        except Exception as e:
            logger.error(f"Could not load training data: {e}")
            return False

        if df.empty or len(df) < 20:
            logger.warning("🧬 Too few memories in KnowledgeBase to evolve (need > 20). Skipping.")
            return False
            
        # Sprawdzamy czy mamy wymagane kolumny
        cols = df.columns
        if 'target' not in cols or ('conf' not in cols and 'confidence' not in cols):
            logger.warning(f"🧬 KnowledgeBase missing required columns for simulation. Valid cols: {cols}")
            return False

        # 2. Initialize Population
        parent_genome = GeneticEvolution.load_genome()
        population = []
        
        # Dodajemy Rodzica (Current Champion)
        population.append(parent_genome)
        
        # Tworzymy 15 pretendentów (Mutants)
        for _ in range(15):
            population.append(GeneticEvolution.mutate(parent_genome))
            
        # 3. Tournament (Obliczanie Fitness)
        best_score = -99999.0
        best_genome = parent_genome
        
        logger.info(f"🧬 Population size: {len(population)}. Fighting on {len(df)} historical scenarios...")
        
        scores = []
        for i, individual in enumerate(population):
            fit_score = GeneticEvolution.calculate_fitness(individual, df)
            scores.append(fit_score)
            
            if fit_score > best_score:
                best_score = fit_score
                best_genome = individual
        
        # 4. Natural Selection
        parent_score = scores[0]
        logger.info(f"🧬 Tournament Results - Parent Score: {parent_score:.2f} | Best Score: {best_score:.2f}")
        
        if best_genome != parent_genome and best_score > parent_score:
            improvement = ((best_score - parent_score) / abs(parent_score)) * 100 if parent_score != 0 else 100
            logger.info(f"🧬 EVOLUTION SUCCESS: New Alpha Genome Found! (+{improvement:.1f}% performance)")
            logger.info(f"🧬 New Config: {json.dumps(best_genome, indent=2)}")
            
            # Zapisz nowego lidera
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(GeneticEvolution.GENOME_FILE), exist_ok=True)
                with open(GeneticEvolution.GENOME_FILE, "w") as f:
                    json.dump(best_genome, f, indent=4)
                return True
            except Exception as e:
                logger.error(f"Failed to save new genome: {e}")
                return False
        else:
            logger.info("🧬 Stagnation: Parent remains dominant. Evolution continues next cycle.")
            return False

if __name__ == "__main__":
    # Test Run
    print("Running Evolution Test...")
    # Mock data if needed or just run
    try:
        updated = GeneticEvolution.run_evolution_cycle()
        print(f"Cycle Complete. Updated: {updated}")
    except Exception as e:
        print(f"Error: {e}")
