from gnn_robustness.model import GCN


def test_gcn_layers_do_not_cache_adjacency_for_inference_corruption():
    model = GCN(input_channels=4, hidden_channels=3, output_channels=2, dropout=0.0)

    assert model.conv1.cached is False
    assert model.conv2.cached is False
