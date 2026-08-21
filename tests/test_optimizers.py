import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dma_pulse.optimizers import DMAPulseOptimizer, SearchParameter


def test_aoa_rfo_pipeline_improves_simple_objective():
    global_space = [SearchParameter("x", -5.0, 5.0)]
    local_space = [SearchParameter("x", -5.0, 5.0)]

    def objective(params):
        return (params["x"] - 1.5) ** 2

    optimizer = DMAPulseOptimizer(global_space, local_space, seed=7)
    global_result, local_result = optimizer.optimize(objective)

    assert local_result.score <= global_result.score
    assert -5.0 <= local_result.params["x"] <= 5.0
