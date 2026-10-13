"""Shared self-test cap. Idle seconds come back unchanged at one load point per cpu or below,
and grow with the 1-minute load per cpu above that. A host that cannot read a load keeps the idle cap.
"""

import math
import os
import subprocess
import sys


def load_cap(idle_s):
    # The regression sets these. The shell caller (`selftest.py 20`) does not.
    # Both names are registry rows owned by this file.
    forced = os.environ.get("BLOOMERY_SELFTEST_LOAD")
    if forced is None:
        try:
            load_avg = os.getloadavg()[0]
        except OSError:
            return idle_s
        cpus = os.cpu_count() or 1
    else:
        load_avg = float(forced)
        ncpu = os.environ.get("BLOOMERY_SELFTEST_NCPU")
        cpus = int(ncpu) if ncpu is not None else (os.cpu_count() or 1)
    # PIN(2026-10-13): scale self-test caps with load per CPU to accommodate CPU contention.
    return idle_s * max(1.0, load_avg / (cpus or 1))


def _constant_factor(idle_s):
    """The mutant: the cap never leaves the idle number."""
    return idle_s


def _sees_ready(policy, idle_s, ready_at):
    """A one-sample wait. Readiness at ready_at seconds is in time when that is at most the cap."""
    return ready_at <= policy(idle_s)


def self_test():
    # Load 80 on 10 cpus, idle 20. Printing the argument alone yields 20, not 160.
    env = os.environ.copy()
    env["BLOOMERY_SELFTEST_LOAD"] = "80"
    env["BLOOMERY_SELFTEST_NCPU"] = "10"
    proc = subprocess.run([sys.executable, os.path.abspath(__file__), "20"],
                          env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "160", proc.stdout
    real_load, real_cpus = os.getloadavg, os.cpu_count
    try:
        os.cpu_count = lambda: 10
        os.getloadavg = lambda: (0.5, 0.5, 0.5)
        assert load_cap(20) == 20, load_cap(20)
        assert load_cap(10) == 10, load_cap(10)
        os.getloadavg = lambda: (10.0, 10.0, 10.0)
        assert load_cap(20) == 20, load_cap(20)
        os.getloadavg = lambda: (80.0, 80.0, 80.0)
        assert load_cap(10) == 80, load_cap(10)
        assert load_cap(20) == 160, load_cap(20)
        # 15 s is past the idle cap of 10 and inside the scaled cap.
        assert _sees_ready(load_cap, 10, 15)
        assert not _sees_ready(_constant_factor, 10, 15), "the constant-factor mutant must miss a readiness past the idle cap"
        os.getloadavg = lambda: (15.0, 15.0, 15.0)
        assert math.ceil(load_cap(20)) == 30, math.ceil(load_cap(20))
        assert math.ceil(_constant_factor(20)) == 20
        os.cpu_count = lambda: None
        os.getloadavg = lambda: (4.0, 4.0, 4.0)
        assert load_cap(10) == 40, load_cap(10)

        def no_load():
            raise OSError("no load")

        os.getloadavg = no_load
        assert load_cap(20) == 20
        assert load_cap(10) == 10
    finally:
        os.getloadavg, os.cpu_count = real_load, real_cpus
    print("selftest: self-test ok")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-test"]:
        raise SystemExit(self_test())
    print(math.ceil(load_cap(float(sys.argv[1]))))
