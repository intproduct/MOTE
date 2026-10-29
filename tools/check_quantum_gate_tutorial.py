"""Reproduce the tutorial's numerical examples without ML dependencies."""

import json
import math
from pathlib import Path


def main():
    results = {}

    def check(name, actual, expected, tolerance=1e-9):
        assert math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), (name, actual, expected)
        results[name] = actual

    check("weighted_output", sum(w * v for w, v in zip([.6, .3, .1], [10, 20, 40])), 16)
    check("top2_output", (6 + 6) / .9, 40 / 3)
    check("weight_transfer", .1 * (20 - 10), 1)
    p = [.6, .3, .1]
    check("amplitude_norm_squared", sum(math.sqrt(v)**2 for v in p), 1)
    for i, v in enumerate(p):
        check(f"softmax_log_p_{i}", math.exp(math.log(v)) / sum(math.exp(math.log(t)) for t in p), v)
    check("sigmoid_gradient", .1 - 1, -.9)
    theta = math.asin(math.sqrt(.1))
    check("angle_gradient", -2 / math.tan(theta), -6)
    check("temperature_first", .6**2 / sum(v*v for v in p), .7826086956521738)
    check("temperature_exercise", .8**2 / (.8**2 + .2**2), 16/17)
    for theta in [math.pi / 6, math.pi / 4]:
        probabilities = [math.cos(theta)**2, math.sin(theta)**2]
        derivative = [-math.sin(2*theta), math.sin(2*theta)]
        check(f"real_fisher_{theta}", sum(d*d/v for d,v in zip(derivative, probabilities)), 4)
    check("rescaled_fisher", 4 * 2**2, 16)
    check("bernoulli_fisher_half", 1/(.5*.5), 4)
    check("bernoulli_fisher_rare", 1/(.01*.99), 101.01010101010101)
    check("local_kl_half", .5*4*.0001**2, 2e-8)
    check("local_kl_rare", .5/(.01*.99)*.0001**2, 5.05050505050505e-7)
    check("natural_gradient_first", .1/100, .001)
    check("preserved_gap", (1-.004)-(.99+.004), .002)
    check("flipped_gap", (1-.006)-(.99+.006), -.002)
    check("many_small_switches", 100*.1**2, 1)
    check("one_large_switch", 20**2, 400)
    check("switch_mse_a", 20*.2**2/1000, .0008)
    check("switch_mse_b", 2*3**2/1000, .018)
    check("ewc_optimum", 2/(1+9), .2)
    check("ewc_old_loss", .2**2/2, .02)
    check("ewc_new_loss", (.2-2)**2/2, 1.62)
    check("ewc_exercise", 2/(1+9/9), 1)
    check("recurrent_contraction", .9**100, .000026561398887587544)
    x = .7
    check("single_ry", math.sin(x/2)**2, (1-math.cos(x))/2)
    check("double_ry", math.sin(x)**2, (1-math.cos(2*x))/2)
    check("curvature_label_one", 1-1/(1+math.exp(-1)), .2689414213699951)
    check("curvature_label_zero", -1/(1+math.exp(-1)), -.7310585786300049)
    check("latency_reduction_percent", 100*(88.727-66.998)/88.727, 24.48972798449175, 1e-5)
    check("ppl_ratio_of_means_percent", 100*(80.39/79.48-1), 1.145, 1e-4)
    check("mlp_switch_percent", 100*339/196608, .17242431640625)
    check("shared_switch_percent", 100*473/196608, .24058024088541667)
    check("margin_reduction_percent", 100*(339-302)/339, 10.914454277286136)
    check("factorial_interaction", (79-62)-(78-60), -1)
    check("final_average", (.6+.72+.7)/3, .6733333333333333)
    check("average_forgetting", ((.8-.6)+(.75-.72))/2, .115)
    check("ppl_ce_exercise_percent", 100*math.expm1(.01), 1.0050167084168058)
    check("switch_exercise_percent", 100*300/200000, .15)
    check("time_a", .2*1000, 200)
    check("time_b", .5*500, 250)
    target = Path(__file__).resolve().parents[1] / "research/quantum_gate_review_20261029/QuantumGate_tutorial_checks.json"
    target.write_text(json.dumps({"passed": len(results), "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(results)} numerical quantities")


if __name__ == "__main__":
    main()
