from flwr.server.strategy import Strategy

class DeterministicAggregationWrapper(Strategy):
    """A wrapper for Flower strategies to ensure deterministic aggregation by sorting client results."""
    
    def __init__(self, base_strategy):
        self.base_strategy = base_strategy

    def aggregate_fit(self, server_round, results, failures):
        # Sort results by client_id to ensure deterministic aggregation
        sorted_results = sorted(
            results,
            key=lambda r: r[1].metrics.get("client_id", "")
        )
        return self.base_strategy.aggregate_fit(server_round, sorted_results, failures)

    def initialize_parameters(self, client_manager):
        return self.base_strategy.initialize_parameters(client_manager)

    def configure_fit(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(self, server_round, parameters, client_manager):
        return self.base_strategy.configure_evaluate(server_round, parameters, client_manager)

    def aggregate_evaluate(self, server_round, results, failures):
        # Sort results by client_id to ensure deterministic aggregation

        sorted_results = sorted(
            results,
            key=lambda r: r[1].metrics.get("client_id", "")
        )
        return self.base_strategy.aggregate_evaluate(server_round, sorted_results, failures)

    def evaluate(self, server_round, parameters):
        return self.base_strategy.evaluate(server_round, parameters)
