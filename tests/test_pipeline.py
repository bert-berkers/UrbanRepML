import unittest
from pathlib import Path
from urban_embedding.pipeline import UrbanEmbeddingPipeline

class TestUrbanEmbeddingPipeline(unittest.TestCase):

    def setUp(self):
        self.project_dir = Path(__file__).parent.parent
        self.city_name = "south_holland_threshold80"
        self.cache_dir = self.project_dir / "cache"
        self.pipeline = UrbanEmbeddingPipeline(
            project_dir=self.project_dir,
            city_name=self.city_name,
            cache_dir=self.cache_dir,
            debug=False
        )
        self.config = self.pipeline.create_default_config(self.city_name)

    def test_pipeline_run(self):
        """Test if the pipeline runs without errors."""
        try:
            embeddings = self.pipeline.run(self.config)
            self.assertIsInstance(embeddings, dict)
            for res, emb in embeddings.items():
                self.assertIsInstance(res, int)
                self.assertIsNotNone(emb)
        except Exception as e:
            self.fail(f"Pipeline run failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
