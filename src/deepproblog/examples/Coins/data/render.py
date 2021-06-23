import argparse
import os
import subprocess

parser = argparse.ArgumentParser(
    description="Render the image data for a given csv file."
)
parser.add_argument("set", nargs="+")
parser.add_argument("-b", "--blender_path", default=None)

if __name__ == "__main__":
    parsed = parser.parse_args()
    blender_path = parsed.blender_path
    if blender_path is None:
        blender_path = "blender"
    for s in parsed.set:
        print("Rendering ", s)

        path = os.path.dirname(os.path.abspath(__file__))
        res = 512
        subprocess.call(
            [
                blender_path,
                path + "/blender_files/scene.blend1",
                "-b",
                "-P",
                path + "/blender_files/render_script.py",
                "--",
                path,
                s,
                str(res),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
