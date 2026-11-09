import pytest
from pytest_celery import RESULT_TIMEOUT, CeleryTestSetup, RedisTestBroker
from pytest_docker_tools.wrappers.container import wait_for_callable

from celery import Celery
from celery.canvas import chain, group
from t.smoke.tasks import long_running_task, noop

WORKER_PREFETCH_MULTIPLIER = 2
WORKER_CONCURRENCY = 5
MAX_PREFETCH = WORKER_PREFETCH_MULTIPLIER * WORKER_CONCURRENCY
PREFETCH_REDUCTION_SKIPPED_MESSAGE = "Skipping prefetch count reduction after connection restart"


def _wait_for_log_count(worker, log: str, count: int, timeout: int = RESULT_TIMEOUT) -> None:
    """Wait until a worker log message appears at least count times."""
    wait_for_callable(
        message=f"Waiting for worker to log '{log}' at least {count} times",
        func=lambda: worker.logs().count(log) >= count,
        timeout=timeout,
    )


def uses_global_qos(celery_setup: CeleryTestSetup) -> bool:
    # Same check as Tasks.qos_global(); Kombu reports per-consumer QoS for
    # RabbitMQ 4, and the worker skips the reduction in that mode (#9512).
    with celery_setup.app.connection_for_read() as connection:
        return not connection.qos_semantics_matches_spec


@pytest.fixture
def default_worker_app(default_worker_app: Celery) -> Celery:
    app = default_worker_app
    app.conf.worker_prefetch_multiplier = WORKER_PREFETCH_MULTIPLIER
    app.conf.worker_concurrency = WORKER_CONCURRENCY
    app.conf.visibility_timeout = 3600
    if app.conf.broker_url.startswith("redis"):
        app.conf.broker_transport_options = {
            "visibility_timeout": app.conf.visibility_timeout,
            "polling_interval": 1,
        }
    if app.conf.result_backend.startswith("redis"):
        app.conf.result_backend_transport_options = {
            "visibility_timeout": app.conf.visibility_timeout,
            "polling_interval": 1,
        }
    return app


class test_worker_enable_prefetch_count_reduction_true:
    @pytest.fixture
    def default_worker_app(self, default_worker_app: Celery) -> Celery:
        app = default_worker_app
        app.conf.worker_enable_prefetch_count_reduction = True
        return app

    @pytest.mark.parametrize("expected_running_tasks_count", range(1, WORKER_CONCURRENCY + 1))
    def test_reducing_prefetch_count(self, celery_setup: CeleryTestSetup, expected_running_tasks_count: int):
        if isinstance(celery_setup.broker, RedisTestBroker):
            # When running in debug it works, when running from CLI it sometimes works
            pytest.xfail("Test is flaky with Redis broker")
        sig = group(long_running_task.s(420) for _ in range(expected_running_tasks_count))
        sig.apply_async(queue=celery_setup.worker.worker_queue)
        celery_setup.broker.restart()

        if not uses_global_qos(celery_setup):
            celery_setup.worker.assert_log_exists(PREFETCH_REDUCTION_SKIPPED_MESSAGE)
            return

        expected_reduced_prefetch = max(
            WORKER_PREFETCH_MULTIPLIER, MAX_PREFETCH - expected_running_tasks_count * WORKER_PREFETCH_MULTIPLIER
        )

        expected_prefetch_reduce_message = (
            f"Temporarily reducing the prefetch count to {expected_reduced_prefetch} "
            f"to avoid over-fetching since {expected_running_tasks_count} tasks are currently being processed."
        )
        celery_setup.worker.assert_log_exists(expected_prefetch_reduce_message)

        expected_prefetch_restore_message = (
            f"The prefetch count will be gradually restored to {MAX_PREFETCH} as the tasks complete processing."
        )
        celery_setup.worker.assert_log_exists(expected_prefetch_restore_message)

    def test_prefetch_count_restored(self, celery_setup: CeleryTestSetup):
        if isinstance(celery_setup.broker, RedisTestBroker):
            # When running in debug it works, when running from CLI it sometimes works
            pytest.xfail("Test is flaky with Redis broker")
        expected_running_tasks_count = MAX_PREFETCH * WORKER_PREFETCH_MULTIPLIER
        sig = group(long_running_task.s(10) for _ in range(expected_running_tasks_count))
        sig.apply_async(queue=celery_setup.worker.worker_queue)
        celery_setup.broker.restart()

        if not uses_global_qos(celery_setup):
            celery_setup.worker.assert_log_exists(PREFETCH_REDUCTION_SKIPPED_MESSAGE)
            return

        expected_prefetch_restore_message = (
            f"Resuming normal operations following a restart.\n"
            f"Prefetch count has been restored to the maximum of {MAX_PREFETCH}"
        )
        celery_setup.worker.assert_log_exists(expected_prefetch_restore_message)

    class test_cancel_tasks_on_connection_loss:
        @pytest.fixture
        def default_worker_app(self, default_worker_app: Celery) -> Celery:
            app = default_worker_app
            app.conf.worker_prefetch_multiplier = 2
            app.conf.worker_cancel_long_running_tasks_on_connection_loss = True
            app.conf.task_acks_late = True
            return app

        def test_max_prefetch_passed_on_broker_restart(self, celery_setup: CeleryTestSetup):
            if isinstance(celery_setup.broker, RedisTestBroker):
                # When running in debug it works, when running from CLI it sometimes works
                pytest.xfail("Test is flaky with Redis broker")
            sig = group(long_running_task.s(420) for _ in range(WORKER_CONCURRENCY))
            sig.apply_async(queue=celery_setup.worker.worker_queue)
            celery_setup.broker.restart()
            noop.s().apply_async(queue=celery_setup.worker.worker_queue)
            celery_setup.worker.assert_log_exists("Task t.smoke.tasks.noop")


class test_worker_enable_prefetch_count_reduction_false:
    @pytest.fixture
    def default_worker_app(self, default_worker_app: Celery) -> Celery:
        app = default_worker_app
        app.conf.worker_prefetch_multiplier = 1
        app.conf.worker_enable_prefetch_count_reduction = False
        app.conf.worker_cancel_long_running_tasks_on_connection_loss = True
        app.conf.task_acks_late = True
        return app

    def test_max_prefetch_not_passed_on_broker_restart(self, celery_setup: CeleryTestSetup):
        if isinstance(celery_setup.broker, RedisTestBroker):
            # When running in debug it works, when running from CLI it sometimes works
            pytest.xfail("Test is flaky with Redis broker")
        sig = group(long_running_task.s(10) for _ in range(WORKER_CONCURRENCY))
        r = sig.apply_async(queue=celery_setup.worker.worker_queue)
        celery_setup.broker.restart()
        noop.s().apply_async(queue=celery_setup.worker.worker_queue)
        assert "Task t.smoke.tasks.noop" not in celery_setup.worker.logs()
        r.get(timeout=RESULT_TIMEOUT)
        assert "Task t.smoke.tasks.noop" in celery_setup.worker.logs()


class test_redis_prefetched_reconnect:
    @pytest.fixture
    def default_worker_app(self, default_worker_app: Celery) -> Celery:
        app = default_worker_app
        app.conf.worker_concurrency = 2
        app.conf.worker_prefetch_multiplier = 2
        return app

    def test_worker_survives_redis_restart_with_prefetched_task(
        self,
        celery_setup: CeleryTestSetup,
    ):
        if not isinstance(celery_setup.broker, RedisTestBroker):
            pytest.skip("This regression requires the Redis broker")

        queue = celery_setup.worker.worker_queue
        assert noop.s().apply_async(queue=queue).get(timeout=RESULT_TIMEOUT) is None

        # Two tasks occupy the pool while the third is prefetched and remains
        # unacknowledged in the worker.
        group(long_running_task.s(20) for _ in range(3)).apply_async(queue=queue)
        _wait_for_log_count(celery_setup.worker, "Starting long running task", 2)

        reserved = celery_setup.app.control.inspect(timeout=RESULT_TIMEOUT).reserved() or {}
        assert sum(len(tasks) for tasks in reserved.values()) >= 1

        # Restart Redis in place. Keeping the outage short matters here: the
        # second close runs while Redis is coming back, which exercises the
        # partially collected transport state from #11054.
        celery_setup.broker.restart()

        # A cleanup failure must not terminate the worker before it reconnects.
        assert noop.s().apply_async(queue=queue).get(timeout=RESULT_TIMEOUT) is None


class test_consumer:
    def test_worker_consume_tasks_after_redis_broker_restart(
        self,
        celery_setup: CeleryTestSetup,
    ):
        queue = celery_setup.worker.worker_queue
        assert noop.s().apply_async(queue=queue).get(timeout=RESULT_TIMEOUT) is None
        celery_setup.broker.kill()
        celery_setup.worker.wait_for_log("Trying again in 8.00 seconds... (4/100)")
        celery_setup.broker.restart()

        count = 5
        assert (
            group(noop.s() for _ in range(count))
            .apply_async(queue=queue)
            .get(timeout=RESULT_TIMEOUT)
            == [None] * count
        )

        assert (
            chain(
                group(noop.si() for _ in range(count)),
                group(noop.si() for _ in range(count)),
            )
            .apply_async(queue=queue)
            .get(timeout=RESULT_TIMEOUT)
            == [None] * count
        )
