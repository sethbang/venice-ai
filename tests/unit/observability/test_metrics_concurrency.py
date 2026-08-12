import threading

import venice_ai.observability.metrics as metrics_module
from venice_ai.observability.metrics import get_enhanced_metrics


class TestMetricsConcurrency:
    def test_singleton_thread_safety(self):
        """Test that get_enhanced_metrics returns the same instance across multiple threads."""
        # Reset singleton
        metrics_module._enhanced_metrics = None

        instances = []
        errors = []

        def get_instance():
            try:
                instance = get_enhanced_metrics()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(20):
            t = threading.Thread(target=get_instance)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert not errors, f"Threads encountered errors: {errors}"
        assert len(instances) == 20

        # Verify all instances are identical
        first_instance = instances[0]
        for i, instance in enumerate(instances[1:]):
            assert instance is first_instance, (
                f"Instance at index {i + 1} is different from the first instance"
            )
