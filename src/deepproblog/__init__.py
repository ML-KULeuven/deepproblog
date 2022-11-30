from pathlib import Path


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            import pytest
            from pathlib import Path
            test_dir = Path(__file__).parents[2] / 'tests/'
            pytest.main([test_dir])
