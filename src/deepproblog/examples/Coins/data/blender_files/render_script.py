import random
import sys
from math import pi

import bpy
# noinspection PyUnresolvedReferences,PyUnresolvedReferences
from mathutils import Vector

argv = sys.argv
argv = argv[argv.index("--") + 1 :]

context = bpy.context
scene = context.scene
camera = bpy.data.objects["Camera"]
coin1 = bpy.data.objects["Coin1"]
coin2 = bpy.data.objects["Coin2"]
lamp = bpy.data.objects["Lamp"]

res = int(argv[2])
scene.render.resolution_x = res
scene.render.resolution_y = res

max_offset = 0.8


def render_csv(f_in, f_out):
    with open(f_in) as f:
        data = [l.strip().split(", ") for l in f]
    for i, (c1, c2) in enumerate(data):
        offset = [max_offset * (random.random() - 0.5) for i in range(2)] + [0]
        coin1.delta_location = Vector(offset)
        flip = c1 == "tails"
        coin1.delta_rotation_euler = [pi * flip, 0, random.random() * 2 * pi]

        offset = [max_offset * (random.random() - 0.5) for i in range(2)] + [0]
        coin2.delta_location = Vector(offset)
        flip = c2 == "tails"
        coin2.delta_rotation_euler = [pi * flip, 0, random.random() * 2 * pi]

        scene.camera = camera
        bpy.data.scenes["Scene"].render.filepath = f_out + "{}.png".format(i)
        bpy.ops.render.render(write_still=True)


render_csv(
    "{}/label_data/{}.csv".format(argv[0], argv[1]),
    "{}/image_data/{}/".format(argv[0], argv[1]),
)
