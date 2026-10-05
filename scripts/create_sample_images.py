from PIL import Image, ImageDraw
import os
os.makedirs('sample_images', exist_ok=True)
# skin tone RGB samples (approx)
skin_tones = [ (244, 219, 196), (214, 172, 139), (180, 134, 100), (140, 85, 60) ]
for i, tone in enumerate(skin_tones):
    img = Image.new('RGB', (256,256), (120,140,180))
    draw = ImageDraw.Draw(img)
    # draw centered oval with skin tone
    draw.ellipse((48,40,208,200), fill=tone)
    img.save(os.path.join('sample_images', f'skin_{i+1}.jpg'))
# add one image with smaller face
img = Image.new('RGB', (256,256), (200,200,200))
draw = ImageDraw.Draw(img)
draw.ellipse((80,80,170,170), fill=(200,160,140))
img.save(os.path.join('sample_images', 'skin_small.jpg'))
print('Wrote sample images to sample_images/')
