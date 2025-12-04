"""Small test runner to execute test functions without pytest available.

This helper will import any module in the tests/ dir named test_*.py and run
any callable starting with 'test_'. It's intentionally lightweight so CI
environments without pytest can still run the unit checks.
"""
import importlib.util
import inspect
import pathlib
import sys


def run_tests():
    tests_dir = pathlib.Path(__file__).parent
    failures = 0
    # Ensure project root is on sys.path so tests can import package modules
    sys.path.insert(0, str(tests_dir.parent))
    for path in tests_dir.glob('test_*.py'):
        name = path.stem
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for n, fn in inspect.getmembers(module, inspect.isfunction):
            if n.startswith('test_'):
                try:
                    fn()
                    print(f"[PASS] {name}.{n}")
                except Exception as e:
                    failures += 1
                    print(f"[FAIL] {name}.{n} -> {e}")

    if failures:
        print(f"\n{failures} tests failed.")
        sys.exit(2)
    print('\nAll tests passed')


if __name__ == '__main__':
    run_tests()
