from find_angle import find_angle

IMAGE_PATH = 'test_images'

passes = 0
fails = 0

images = { }
with open(IMAGE_PATH + '/list.txt') as f:
    lines = f.readlines()
    for line in lines:
        frags = line.split(',')
        images[frags[0]] = frags[1]

for image, angle in images.items():
    path = IMAGE_PATH + '/' + image
    theta = find_angle(path)

    angle = round(float(angle))
    theta = round(float(theta))

    print(image)
    print(f'Expected angle: {angle}°')
    print(f'Found angle: {theta}°')
    if angle == theta:
        print('TEST PASSED\n')
        passes += 1
    else:
        print('TEST FAILED\n')
        fails += 1
    
print(f'PASSES: {passes} | FAILS: {fails}')