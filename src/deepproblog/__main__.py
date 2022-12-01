from pathlib import Path


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == 'test':
            import pytest
            from pathlib import Path
            print(Path(__file__).parent)
            pytest.main([str(Path(__file__).parent)])
