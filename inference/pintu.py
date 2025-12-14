from PIL import Image, ImageDraw, ImageFont
import os


def concatenate_images_horizontally(folders, image_name, output_path):
    # 存储打开的图片
    images = []

    for folder in folders:
        try:
            if "snp" in folder:
                s = image_name[:-4] + "_final.png"
            else:
                s = image_name
            img_path = f"{folder}/{s}"
            img = Image.open(img_path)
            img = img.resize((512, 512))
            images.append(img)
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
            continue

    # 检查是否有图片被成功加载
    if not images:
        print("没有图片可拼接")
        return

    total_width = sum(i.width for i in images)
    max_height = max(i.height for i in images)

    new_img = Image.new('RGB', (total_width, max_height), color='white')
    x_offset = 0
    draw = ImageDraw.Draw(new_img)

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except IOError:
        font = None

    for idx, img in enumerate(images):
        new_img.paste(img, (x_offset, 0))

        text = folders[idx].split('/')[-1]
        try:
            if font is not None:
                text_width, text_height = font.getsize(text)
            else:
                text_width, text_height = 50, 20  # 默认值
        except AttributeError:
            text_width, text_height = 50, 20

        text_x = x_offset + (img.width - text_width) // 2

        draw.text((text_x, max_height - text_height - 5), text, fill="black", font=font)

        x_offset += img.width

    new_img.save(output_path)


# 示例用法
folders = ['/local/lly/StrokeTest/snp/output_500/',
           '/local/lly/StrokeTest/PT/pt-256-1bianxie/',
           '/local/lly/StrokeTest/i2o/p81-689/imgs3/',
           '/local/lly/StrokeTest/cnp/output500/',
           '/local/lly/StrokeTest/ours/output_256_xie1bian/',
           '/local/lly/StrokeTest/l2p/round100+80*5/',
            '/local/lly/StrokeTest/ours/output_256_c_t_xie1bian/',
           ]  # 500

folders = ['/local/lly/StrokeTest/snp/output_1000/',
           '/local/lly/StrokeTest/PT/output_256/',
           '/local/lly/StrokeTest/i2o/p49-972/imgs3/',
           '/local/lly/StrokeTest/cnp/output1000/',
           '/local/lly/StrokeTest/ours/output_256/',
           '/local/lly/StrokeTest/l2p/round1000-12*80/',
           '/local/lly/StrokeTest/ours/output_256_c_t/',
           ]  # 1000
folders = ['/local/lly/StrokeTest/snp/output_3000/',
           '/local/lly/StrokeTest/PT/pt-512-1bianxie/',
           '/local/lly/StrokeTest/i2o/p16/imgs3/',
           '/local/lly/StrokeTest/cnp/output3000/',
           '/local/lly/StrokeTest/ours/output_512_xie1bian/',
           '/local/lly/StrokeTest/l2p/round3000-36*80/',
           '/local/lly/StrokeTest/ours/output_512_c_t_xie1bian/',
           ]  # 3000
pics = os.listdir("/home/lly/workspace/PaintTransformer-main/imgs3/")
for pic in pics:
    image_name = pic  # 图片名
    print(pic)
    output_path = "/local/lly/StrokeTest/pintu-3000/"  # 输出路径
    os.makedirs(output_path, exist_ok=True)
    concatenate_images_horizontally(folders, image_name, output_path + pic)
