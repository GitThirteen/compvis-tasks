import os
import time
from find_angle import find_angle

IMAGE_PATH = os.path.join(os.path.dirname(__file__), '../data/task_1/test_images')

passes = 0
fails = 0

images = { }
with open(IMAGE_PATH + '/list.txt') as f:
    lines = f.readlines()
    for line in lines:
        frags = line.split(',')
        images[frags[0]] = frags[1]

start_time = time.time()
for name, angle in images.items():
    indiv_start_time = time.time()
    path = IMAGE_PATH + '/' + name
    theta = find_angle(path)

    angle = round(float(angle))
    theta = round(float(theta))

    print(name)
    print(f'Expected angle: {angle}°')
    print(f'Found angle: {theta}°')
    if angle == theta:
        print('TEST PASSED')
        passes += 1
    else:
        print('TEST FAILED')
        fails += 1

    indiv_end_time = round(time.time() - indiv_start_time, 3)
    print(f'RUNTIME: {indiv_end_time}s\n')

end_time = round(time.time() - start_time, 3)
print(f'PASSES: {passes} | FAILS: {fails} | RUNTIME: {end_time}s')
