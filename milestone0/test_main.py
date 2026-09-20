from pathlib import Path


source = Path(__file__).with_name("main.py")
assert source.exists(), "main.py is missing"
contents = source.read_text(encoding="utf-8")
assert "def main()" in contents
assert "mp.tasks.vision.HandLandmarker" in contents
assert 'if __name__ == "__main__":' in contents
