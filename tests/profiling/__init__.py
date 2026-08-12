"""
Performance profiling tests for Venice AI.

These tests measure performance characteristics and help identify
optimization opportunities. They are not part of the regular test suite.

Run profiling tests with:
    poetry run pytest tests/profiling/ -v --tb=short

Individual profiling suites:
    poetry run pytest tests/profiling/test_redis_performance.py -v
    poetry run pytest tests/profiling/test_connection_pooling.py -v

KNOWN ISSUE:
    You may see asyncio warnings like:
        "Task was destroyed but it is pending!"
        task: <Task pending name='Task-XXX' coro=<ConnectionPool.disconnect()..."

    This is a harmless cleanup warning from redis-py's background disconnect tasks.
    The tests pass successfully and the warning can be safely ignored.
    This does not affect test functionality or results.
"""
