from __future__ import annotations

import unittest

import torch

from MOTE.train.rl_controller import compute_post_update_policy_drift


class PostUpdatePolicyDiagnosticsTests(unittest.TestCase):
    def test_reports_live_ratio_and_clip_only_on_response_tokens(self):
        old = torch.zeros((2, 4), dtype=torch.float32)
        post = torch.tensor(
            [
                [9.0, 0.0, 0.1, 0.3],
                [-9.0, 0.0, -0.1, -0.3],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=torch.bool,
        )

        metrics = compute_post_update_policy_drift(post, old, mask, eps_clip=0.2)

        expected = torch.exp(torch.tensor([0.1, 0.3, -0.1, -0.3]))
        self.assertAlmostEqual(metrics["post_update_ratio_mean"], float(expected.mean()), places=6)
        self.assertAlmostEqual(metrics["post_update_ratio_max"], float(torch.exp(torch.tensor(0.3))), places=6)
        self.assertAlmostEqual(metrics["post_update_raw_log_ratio_max"], 0.3, places=6)
        self.assertAlmostEqual(metrics["post_update_clip_frac"], 0.5, places=6)

    def test_rejects_shape_mismatch_and_empty_response(self):
        values = torch.zeros((1, 2))
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            compute_post_update_policy_drift(values, torch.zeros((1, 3)), values, 0.2)
        with self.assertRaisesRegex(RuntimeError, "no response tokens"):
            compute_post_update_policy_drift(values, values, torch.zeros_like(values), 0.2)


if __name__ == "__main__":
    unittest.main()
