import unittest
import os
import shutil
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

# Add project root to sys.path to allow direct import of project modules
import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from preprocess_data import preprocess_data, PreprocessingStats # Assuming PreprocessingStats is still relevant
from models import DKT
from main import get_paf_output_dimensions # Assuming main.py has this function
from config import PATHS # For default paths, might need to override for tests

# Define a temporary directory for test artifacts
TEST_CHECKPOINTS_DIR = Path("tests/tmp_checkpoints")
TEST_PAF_OUTPUTS_DIR = TEST_CHECKPOINTS_DIR / "paf_outputs"
TEST_DATA_DIR = Path("tests/tmp_data")
DUMMY_INPUT_CSV = TEST_DATA_DIR / "dummy_input_data.csv"
DUMMY_OUTPUT_CSV = TEST_DATA_DIR / "dummy_preprocessed_data.csv"

# Expected number of factors for PAF (must match preprocess_data.py)
# Or make NUM_FACTORS in preprocess_data.py configurable and pass it here.
# For now, assume it's hardcoded as 10 in preprocess_data.py
EXPECTED_NUM_FACTORS = 10


class TestPAFPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        TEST_PAF_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Create dummy CSV data
        cls.dummy_data = {
            'user_id': [1, 1, 2, 2, 3, 3, 1, 2, 4, 4, 4],
            'skill_id': [101, 102, 101, 103, 102, 103, 101, 103, 104, 104, 101], # 4 unique skills
            'correct': [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0],
            'timestamp': pd.to_datetime([
                '2023-01-01 10:00:00', '2023-01-01 10:01:00', '2023-01-01 10:02:00',
                '2023-01-01 10:03:00', '2023-01-01 10:04:00', '2023-01-01 10:05:00',
                '2023-01-02 11:00:00', '2023-01-02 11:01:00', '2023-01-03 12:00:00',
                '2023-01-03 12:01:00', '2023-01-03 12:02:00'
            ])
        }
        cls.dummy_df = pd.DataFrame(cls.dummy_data)
        cls.dummy_df.to_csv(DUMMY_INPUT_CSV, index=False)
        cls.unique_skills_in_dummy_data = sorted(cls.dummy_df['skill_id'].unique())


    @classmethod
    def tearDownClass(cls):
        if TEST_CHECKPOINTS_DIR.exists():
            shutil.rmtree(TEST_CHECKPOINTS_DIR)
        if TEST_DATA_DIR.exists():
            shutil.rmtree(TEST_DATA_DIR)

    def test_paf_processing_runs_and_creates_outputs(self):
        # Override PATHS.checkpoint_dir for this test if preprocess_data uses it directly
        # This is a bit tricky. preprocess_data.py writes to Path('checkpoints') hardcoded in some places.
        # For a true unit test, preprocess_data.py should be refactored to accept output paths.
        # Given the current structure, we'll check the default 'checkpoints' path after running,
        # or ideally, mock Path('checkpoints') to point to our TEST_CHECKPOINTS_DIR.
        
        # For now, let's assume preprocess_data will write to a global "checkpoints" directory.
        # We will copy those to our test dir for assertion and then clean up.
        # A better way is to make preprocess_data accept an output_base_dir.
        
        # Let's try to patch the Path constructor for 'checkpoints'
        original_path_init = Path.__init__
        def mocked_path_init(self, *args, **kwargs):
            if args and args[0] == 'checkpoints':
                original_path_init(self, TEST_CHECKPOINTS_DIR, **kwargs)
            else:
                original_path_init(self, *args, **kwargs)
        
        with unittest.mock.patch('pathlib.Path.__init__', mocked_path_init):
            preprocess_data(str(DUMMY_INPUT_CSV), str(DUMMY_OUTPUT_CSV))

        # Assert output files are created in TEST_PAF_OUTPUTS_DIR
        self.assertTrue((TEST_PAF_OUTPUTS_DIR / "skill_factor_embeddings.npy").exists())
        self.assertTrue((TEST_PAF_OUTPUTS_DIR / "original_skill_id_to_factor_idx.json").exists())
        self.assertTrue((TEST_PAF_OUTPUTS_DIR / "factor_idx_to_original_skill_id.json").exists())
        self.assertTrue((TEST_PAF_OUTPUTS_DIR / "paf_model.joblib").exists())
        self.assertTrue(DUMMY_OUTPUT_CSV.exists())
        self.assertTrue((TEST_CHECKPOINTS_DIR / "preprocessing_summary.json").exists())

    def test_paf_embedding_dimensions(self):
        # Similar mocking strategy as above for Path
        original_path_init = Path.__init__
        def mocked_path_init(self, *args, **kwargs):
            if args and args[0] == 'checkpoints':
                original_path_init(self, TEST_CHECKPOINTS_DIR, **kwargs)
            else:
                original_path_init(self, *args, **kwargs)
        
        with unittest.mock.patch('pathlib.Path.__init__', mocked_path_init):
            preprocess_data(str(DUMMY_INPUT_CSV), str(DUMMY_OUTPUT_CSV))

        embeddings = np.load(TEST_PAF_OUTPUTS_DIR / "skill_factor_embeddings.npy")
        
        # Number of unique skills in our dummy data that have interactions.
        # Skill 104 has interactions from user 4. Skill 101, 102, 103 have interactions.
        # All 4 unique skills (101, 102, 103, 104) should be in the user-skill matrix if they have interactions.
        # The dummy data has 4 unique skills: 101, 102, 103, 104. All appear in interactions.
        expected_num_unique_skills = len(self.unique_skills_in_dummy_data)
        
        self.assertEqual(embeddings.shape, (expected_num_unique_skills, EXPECTED_NUM_FACTORS))

    def test_paf_skill_id_mapping(self):
        original_path_init = Path.__init__
        def mocked_path_init(self, *args, **kwargs):
            if args and args[0] == 'checkpoints':
                original_path_init(self, TEST_CHECKPOINTS_DIR, **kwargs)
            else:
                original_path_init(self, *args, **kwargs)
        
        with unittest.mock.patch('pathlib.Path.__init__', mocked_path_init):
            preprocess_data(str(DUMMY_INPUT_CSV), str(DUMMY_OUTPUT_CSV))

        with open(TEST_PAF_OUTPUTS_DIR / "original_skill_id_to_factor_idx.json", 'r') as f:
            skill_to_idx_map = json.load(f)

        # Convert keys from string back to int if they were stored as int originally
        skill_to_idx_map_int_keys = {int(k): v for k, v in skill_to_idx_map.items()}

        self.assertSetEqual(set(skill_to_idx_map_int_keys.keys()), set(self.unique_skills_in_dummy_data))
        
        factor_indices = list(skill_to_idx_map_int_keys.values())
        self.assertTrue(all(0 <= idx < len(self.unique_skills_in_dummy_data) for idx in factor_indices))
        self.assertEqual(len(set(factor_indices)), len(self.unique_skills_in_dummy_data)) # All indices unique


class TestDKTWithPAFEmbeddings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        TEST_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure base test dir exists
        TEST_PAF_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True) # Ensure paf_outputs for dummy embedding

        cls.num_dummy_skills = 5
        cls.num_dummy_factors = 3
        cls.dummy_embeddings_array = np.random.rand(cls.num_dummy_skills, cls.num_dummy_factors).astype(np.float32)
        cls.dummy_embeddings_path = TEST_PAF_OUTPUTS_DIR / "dummy_skill_factor_embeddings.npy"
        np.save(cls.dummy_embeddings_path, cls.dummy_embeddings_array)

    @classmethod
    def tearDownClass(cls):
        if cls.dummy_embeddings_path.exists():
            os.remove(cls.dummy_embeddings_path)
        # Only remove paf_outputs if it's empty, or manage cleanup more carefully if shared
        if TEST_PAF_OUTPUTS_DIR.exists() and not os.listdir(TEST_PAF_OUTPUTS_DIR):
             shutil.rmtree(TEST_PAF_OUTPUTS_DIR)
        # Do not remove TEST_CHECKPOINTS_DIR here if other test classes use it.
        # This will be handled by the TestPAFPreprocessing.tearDownClass or a general test runner cleanup.

    def test_dkt_initialization_with_paf(self):
        model = DKT(
            num_outputs=self.num_dummy_skills,
            embedding_dim=self.num_dummy_factors,
            skill_embeddings_path=str(self.dummy_embeddings_path),
            hidden_dim=32 # Small hidden dim for test
        )
        self.assertEqual(model.embedding.weight.shape, (self.num_dummy_skills * 2, self.num_dummy_factors))
        
        # Check if weights are loaded correctly (expanded)
        expected_weights = torch.zeros(self.num_dummy_skills * 2, self.num_dummy_factors)
        loaded_paf_embeddings = torch.tensor(self.dummy_embeddings_array, dtype=torch.float)
        for i in range(self.num_dummy_skills):
            expected_weights[2*i] = loaded_paf_embeddings[i]
            expected_weights[2*i + 1] = loaded_paf_embeddings[i]
        
        self.assertTrue(torch.allclose(model.embedding.weight.data, expected_weights, atol=1e-6))
        self.assertTrue(model.embedding.weight.requires_grad) # Check if fine-tuning is enabled

    def test_dkt_forward_pass_with_paf(self):
        model = DKT(
            num_outputs=self.num_dummy_skills,
            embedding_dim=self.num_dummy_factors,
            skill_embeddings_path=str(self.dummy_embeddings_path),
            hidden_dim=32
        )
        model.eval() # Set to eval mode

        batch_size = 4
        seq_len = 10
        
        # factor_idx should be used for skill_ids
        skill_ids = torch.randint(0, self.num_dummy_skills, (batch_size, seq_len))
        correct = torch.randint(0, 2, (batch_size, seq_len))
        # Create a mask (e.g., all valid for simplicity, or some padded)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        # Simulate some padding for a more robust test
        if seq_len > 5:
            mask[0, seq_len-2:] = 0 # User 0 has shorter sequence
            mask[1, seq_len-1:] = 0 # User 1 has shorter sequence


        # The model expects a dictionary-like batch object with attributes
        class DummyBatch:
            def __init__(self, skill_ids, correct, mask):
                self.skill_ids = skill_ids
                self.correct = correct
                self.mask = mask

        dummy_batch = DummyBatch(skill_ids, correct, mask)
        
        try:
            with torch.no_grad():
                output_logits = model(dummy_batch)
            self.assertEqual(output_logits.shape, (batch_size, seq_len, self.num_dummy_skills))
        except Exception as e:
            self.fail(f"DKT forward pass failed with PAF embeddings: {e}")


class TestMainPAFIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure the base directory for paf_outputs exists for this test class
        cls.temp_checkpoints_dir = Path("tests/tmp_main_checkpoints")
        cls.temp_paf_outputs_dir = cls.temp_checkpoints_dir / "paf_outputs"
        cls.temp_paf_outputs_dir.mkdir(parents=True, exist_ok=True)

        cls.num_main_dummy_skills = 7
        cls.num_main_dummy_factors = 4
        cls.main_dummy_embeddings_array = np.random.rand(
            cls.num_main_dummy_skills, cls.num_main_dummy_factors
        ).astype(np.float32)
        
        cls.main_dummy_embeddings_path = cls.temp_paf_outputs_dir / "main_dummy_skill_factor_embeddings.npy"
        np.save(cls.main_dummy_embeddings_path, cls.main_dummy_embeddings_array)

    @classmethod
    def tearDownClass(cls):
        if cls.main_dummy_embeddings_path.exists():
            os.remove(cls.main_dummy_embeddings_path)
        if cls.temp_paf_outputs_dir.exists(): # remove paf_outputs
            shutil.rmtree(cls.temp_paf_outputs_dir)
        if cls.temp_checkpoints_dir.exists(): # remove base for this test class
            shutil.rmtree(cls.temp_checkpoints_dir)
            
    def test_get_paf_output_dimensions(self):
        # main.get_paf_output_dimensions expects the full path to the .npy file
        num_outputs, embedding_dim = get_paf_output_dimensions(str(self.main_dummy_embeddings_path))
        self.assertEqual(num_outputs, self.num_main_dummy_skills)
        self.assertEqual(embedding_dim, self.num_main_dummy_factors)

    def test_get_paf_output_dimensions_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            get_paf_output_dimensions("non_existent_path.npy")


if __name__ == '__main__':
    # It's good practice to ensure all test classes are run if the file is executed directly
    # However, specific test runs are usually handled by a test runner like `python -m unittest discover`
    # For simplicity here, unittest.main() will discover all TestCases in this file.
    unittest.main()
