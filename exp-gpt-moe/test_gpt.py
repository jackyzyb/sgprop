import unittest

import torch

from model import FFN, GPT, GPTConfig, SGFFN
from sg_module import SyntheticGradientHook
from train import (
    TrainConfig,
    make_optimizer,
    run_sg_phase,
    selected_sg_phase_steps,
    sg_microbatch_size,
    training_objective,
)


def tiny_config(**overrides) -> GPTConfig:
    values = {
        "vocab_size": 32,
        "block_size": 8,
        "n_layer": 2,
        "n_head": 2,
        "d_model": 8,
        "dropout": 0.0,
        "use_sg": False,
    }
    values.update(overrides)
    return GPTConfig(**values)


def tiny_sg_config(**overrides) -> GPTConfig:
    values = {
        "use_sg": True,
        "sg_synthetic_grad": True,
        "sg_num_layers": 1,
        "sg_kwargs": {
            "num_experts": 2,
            "expert_dim": 4,
            "top_k": 1,
            "synthetic_grad_lr": 1e-3,
            "synthetic_grad_lambda": 1.0,
        },
    }
    values.update(overrides)
    return tiny_config(**values)


class GPTSyntheticGradientTest(unittest.TestCase):
    def test_backprop_variants_update_base_parameters(self) -> None:
        for use_sg in (False, True):
            with self.subTest(use_sg=use_sg):
                model = GPT(tiny_sg_config(use_sg=use_sg, sg_synthetic_grad=False))
                self.assertIsInstance(model.transformer["h"][0].ffn, SGFFN if use_sg else FFN)
                self.assertEqual(list(model.synthetic_gradient_parameters()), [])
                before = model.lm_head.weight.detach().clone()
                optimizer = make_optimizer(model, TrainConfig())
                tokens = torch.randint(0, model.config.vocab_size, (2, 6))
                _, loss = model(tokens, tokens)
                loss.backward()
                optimizer.step()
                self.assertFalse(torch.equal(before, model.lm_head.weight))

    def test_attention_is_causal(self) -> None:
        model = GPT(tiny_config()).eval()
        tokens = torch.randint(0, model.config.vocab_size, (2, 6))
        changed = tokens.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % model.config.vocab_size
        with torch.no_grad():
            original, _ = model(tokens)
            modified, _ = model(changed)
        torch.testing.assert_close(original[:, :3], modified[:, :3])


    def test_predictor_parameters_are_excluded_from_base_optimizer(self) -> None:
        model = GPT(tiny_sg_config())
        predictor_ids = {
            id(parameter) for parameter in model.synthetic_gradient_parameters()
        }
        base_ids = {id(parameter) for parameter in model.base_parameters()}

        self.assertTrue(predictor_ids)
        self.assertTrue(predictor_ids.isdisjoint(base_ids))
        self.assertEqual(
            predictor_ids | base_ids,
            {id(parameter) for parameter in model.parameters()},
        )


    def test_sgffn_predictors_train(self) -> None:
        model = GPT(
            tiny_config(
                n_layer=1,
                use_sg=True,
                sg_synthetic_grad=True,
                sg_num_layers=1,
                sg_kwargs={
                    "num_experts": 2,
                    "expert_dim": 4,
                    "top_k": 1,
                    "synthetic_grad_lr": 1e-3,
                },
            )
        )
        input_ids = torch.randint(0, model.config.vocab_size, (2, 6))
        targets = torch.randint(0, model.config.vocab_size, (2, 6))

        _, loss = model(input_ids, targets)
        loss.backward()
        sg_loss, _ = model.update_sg_modules()

        self.assertGreater(sg_loss.item(), 0.0)

    def test_convex_gradient_mixing(self) -> None:
        target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        prediction = torch.tensor([[0.0, 2.0], [-2.0, 0.0]])

        for weight in (0.0, 0.25, 1.0):
            with self.subTest(weight=weight):
                output = torch.ones_like(target, requires_grad=True)
                hook = SyntheticGradientHook(weight)
                hook.register(output, prediction)
                output.backward(target)
                torch.testing.assert_close(
                    output.grad, weight * target + (1.0 - weight) * prediction,
                )
                torch.testing.assert_close(hook.target, target)

    def test_same_sign_fraction_ignores_zero_targets(self) -> None:
        output = torch.ones(1, 4, requires_grad=True)
        prediction = torch.tensor([[2.0, -3.0, 0.0, 4.0]], requires_grad=True)
        target = torch.tensor([[1.0, -1.0, -2.0, 0.0]])
        hook = SyntheticGradientHook(1.0)

        hook.register(output, prediction)
        output.backward(target)
        _, stats = hook.loss()

        self.assertAlmostEqual(stats["same_sign_fraction"].item(), 2 / 3)


    def test_sg_target_is_not_divided_by_batch_size(self) -> None:
        torch.manual_seed(3)
        model = GPT(
            tiny_sg_config(n_layer=1)
        )
        config = TrainConfig(sg_phase_steps=1, routing_entropy_coef=0.0, load_balance_coef=0.0)
        input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
        targets = torch.randint(0, model.config.vocab_size, (1, 6))

        objective, _, _, _ = training_objective(
            model,
            input_ids,
            targets,
            config,
        )
        objective.backward()
        module = next(model.sg_modules())
        single_target = module.blocks[0].synthetic_gradient_hook.target.clone()

        model.zero_grad(set_to_none=True)
        other_input_ids = torch.randint(0, model.config.vocab_size, (1, 6))
        other_targets = torch.randint(0, model.config.vocab_size, (1, 6))
        objective, _, _, _ = training_objective(
            model,
            torch.cat((input_ids, other_input_ids)),
            torch.cat((targets, other_targets)),
            config,
        )
        objective.backward()
        batched_target = module.blocks[0].synthetic_gradient_hook.target[:6]

        self.assertTrue(torch.allclose(single_target, batched_target, atol=1e-7))

    def test_sg_phase_updates_only_predictors_for_each_microbatch(self) -> None:
        torch.manual_seed(4)
        model = GPT(tiny_sg_config(n_layer=1))
        config = TrainConfig(sg_phase_steps=2, sg_phase_epochs=1)
        input_ids = torch.randint(0, model.config.vocab_size, (4, 6))
        targets = torch.randint(0, model.config.vocab_size, (4, 6))
        base_ids = {id(parameter) for parameter in model.base_parameters()}
        base_before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if id(parameter) in base_ids
        }

        (sg_loss, _), dynamics = run_sg_phase(
            model,
            input_ids,
            targets,
            config,
            capture_dynamics=True,
        )

        self.assertIsNotNone(sg_loss)
        self.assertEqual([row["sg_update"] for row in dynamics], [1, 2])
        self.assertEqual([row["sg_epoch"] for row in dynamics], [1, 1])
        self.assertEqual([row["sg_update_in_epoch"] for row in dynamics], [1, 2])
        self.assertTrue(
            all("same_sign_fraction" in row["sg_stats"] for row in dynamics)
        )
        for name, before in base_before.items():
            self.assertTrue(torch.equal(before, dict(model.named_parameters())[name]))
        module = next(model.sg_modules())
        optimizer_step = next(iter(module.synthetic_grad_optimizer.state.values()))[
            "step"
        ]
        self.assertEqual(optimizer_step.item(), 2)

    def test_multiple_sg_phase_epochs_repeat_predictor_updates(self) -> None:
        torch.manual_seed(5)
        model = GPT(tiny_sg_config(n_layer=1))
        config = TrainConfig(sg_phase_steps=2)
        input_ids = torch.randint(0, model.config.vocab_size, (4, 6))
        targets = torch.randint(0, model.config.vocab_size, (4, 6))

        (sg_loss, _), dynamics = run_sg_phase(
            model,
            input_ids,
            targets,
            config,
            capture_dynamics=True,
        )

        self.assertIsNotNone(sg_loss)
        self.assertEqual([row["sg_update"] for row in dynamics], [1, 2, 3, 4])
        self.assertEqual([row["sg_epoch"] for row in dynamics], [1, 1, 2, 2])
        self.assertEqual(
            [row["sg_update_in_epoch"] for row in dynamics],
            [1, 2, 1, 2],
        )
        module = next(model.sg_modules())
        optimizer_step = next(iter(module.synthetic_grad_optimizer.state.values()))[
            "step"
        ]
        self.assertEqual(optimizer_step.item(), 4)

    def test_one_sg_phase_epoch_matches_the_original_single_pass(self) -> None:
        torch.manual_seed(6)
        model = GPT(tiny_sg_config(n_layer=1))
        torch.manual_seed(6)
        reference = GPT(tiny_sg_config(n_layer=1))
        input_ids = torch.randint(0, model.config.vocab_size, (4, 6))
        targets = torch.randint(0, model.config.vocab_size, (4, 6))
        config = TrainConfig(sg_phase_steps=2, sg_phase_epochs=1)

        run_sg_phase(
            model,
            input_ids,
            targets,
            config,
        )
        reference.set_synthetic_gradient_training(True)
        for start in (0, 2):
            reference.zero_grad(set_to_none=True)
            objective, _, _, _ = training_objective(
                reference,
                input_ids[start : start + 2],
                targets[start : start + 2],
                config,
            )
            objective.backward()
            reference.update_sg_modules()

        actual = next(model.sg_modules())
        expected = next(reference.sg_modules())
        for actual_param, expected_param in zip(
            actual.parameters(),
            expected.parameters(),
        ):
            self.assertTrue(torch.equal(actual_param, expected_param))

    def test_sg_microbatch_size_requires_exact_division(self) -> None:
        self.assertEqual(sg_microbatch_size(16, 8), 2)
        with self.assertRaisesRegex(ValueError, "divisible"):
            sg_microbatch_size(15, 8)

    def test_sg_phase_epochs_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "sg_phase_epochs"):
            TrainConfig(sg_phase_epochs=0)

    def test_five_sg_phases_span_training(self) -> None:
        self.assertEqual(selected_sg_phase_steps(200), (1, 51, 101, 150, 200))
        self.assertEqual(selected_sg_phase_steps(3), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
