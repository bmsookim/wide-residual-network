require 'image'

new_path = 'svhn_image.png'

img = image.load('cifar10_image.png')
new_img = image.load(new_path)

print(img:size())
print(new_img:size())

-- new_img:resizeAs(img)
-- image.save(new_path, new_img)


