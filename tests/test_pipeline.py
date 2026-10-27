import unittest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from src.pipeline import get_data, split_data, make_models, cross_validate_models, select_model, eval_selected


class TestScientificProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, _ = get_data()
        cls.tr, cls.te = split_data(cls.X, cls.y)

    def test_positive_class_is_malignant(self):
        self.assertEqual(int(self.y.sum()), 212)
        self.assertEqual(int((self.y == 0).sum()), 357)
        self.assertEqual(self.X.shape, (569, 30))

    def test_split_is_disjoint_and_reproducible(self):
        t2, e2 = split_data(self.X, self.y)
        np.testing.assert_array_equal(self.tr, t2)
        np.testing.assert_array_equal(self.te, e2)
        self.assertEqual(len(set(self.tr) & set(self.te)), 0)
        self.assertEqual(len(self.tr) + len(self.te), 569)

    def test_all_models_scale_in_pipeline(self):
        models = make_models()
        self.assertEqual(len(models), 4)
        for estimator in models.values():
            self.assertIsInstance(estimator, Pipeline)
            self.assertEqual(estimator.steps[0][0], 'scale')
            if 'pca' in estimator.named_steps:
                self.assertIsInstance(estimator.named_steps['pca'], PCA)

    def test_training_only_evaluation_ranges(self):
        # Cross-validation operates exclusively on training observations.
        models = make_models()
        data = cross_validate_models(models, self.X.iloc[self.tr], self.y.iloc[self.tr])
        self.assertEqual(len(data), 4)
        self.assertTrue(data['cv_auc_mean'].between(0, 1).all())
        self.assertIn(select_model(data), models)


if __name__ == '__main__':
    unittest.main()
